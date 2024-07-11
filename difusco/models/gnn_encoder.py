import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint


'''
这个文件就是对各种特征进行的处理包括时间步以及原本特征还有位置编码
针对稀疏图和非稀疏图的节点特征和边特征进行处理，包括聚合邻居特征、融合位置编码以及整合时间步特征
如果要加傅里叶特征等就在这个文件加
'''

class GNNLayer(nn.Module):
  """Configurable GNN Layer捕捉图的结构信息以及节点和边的特征信息

  现了一个灵活的图神经网络层，能够根据输入的节点特征、边特征和图结构进行节点特征和边特征的更新、聚合和归一化处理，
  支持稠密和稀疏数据的处理，并能够通过门控机制增强模型的表达能力。
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """根据传入的参数选择不同的聚合、归一化方式，并且支持边门控机制，以适应不同类型的图数据分析任务。

    Args:
        hidden_dim: Hidden dimension size (int) 隐藏层维度大小，指定节点特征的维度
        aggregation: Neighborhood aggregation scheme 邻域聚合方案("sum"/"mean"/"max")
        norm: Feature normalization scheme 特征归一化方案 ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False) 是否使用可学习的仿射参数来归一化特征
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False) 是否使用批次统计信息来计算归一化的均值和标准差
        gated: Whether to use edge gating (True/False) 是否使用边门控（edge gating）机制，这对于图卷积网络（GCN）是必需的，因此必须为 True
    """
    super(GNNLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag" # 确保正在执行过程中必须使用边门控制机制

    # 用于更新节点和边特征的线性变换层 输入输出维度都是 hidden_dim 并且使用bias
    self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    # 判断使用怎样的 点 归一化
    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    # 判断使用怎样的 边 归一化
    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h, e, graph, mode="residual", edge_index=None, sparse=False):
    """ 返回的是聚合了的边特征与节点特征 就是更新节点与边的特征为了后续计算，确保节点和边的信息可以在整个网络中流动和更新

    Args:
        In Dense version:（稠密图）
          h: 输入的节点特征矩阵 (B x V x H)  B 是批次大小，V 是节点数，H 是特征维度
          e: 输入的边特征矩阵 (B x V x V x H) B 是批次大小，V x V 是节点之间的边，H 是特征维度。
          graph: 图的邻接矩阵 (B x V x V)
          mode: 模式字符串，决定是否使用残差链接
        In Sparse version:（稀疏图）
          h: 输入的节点特征 (V x H)V 是节点数
          e: 输入的边特征 (E x H) E 是边数
          graph: 使用稀疏张量表示的图结构，类型为 torch_sparse.SparseTensor
          mode: str
          edge_index: Edge indices (2 x E) 边的索引，维度为 (2 x E)，其中每列表示边连接的两个节点的索引
        sparse: 共同参数 代表是否使用稀疏张量表示 (True/False)
    Returns:
        更新后的节点和边的特征
    """
    if not sparse:
      batch_size, num_nodes, hidden_dim = h.shape
    else:
      batch_size = None
      num_nodes, hidden_dim = h.shape
    h_in = h
    e_in = e

    # 节点更新的线性变换
    Uh = self.U(h)  # B x V x H

    if not sparse:
      # 在稠密图表示中，通常会假定图是完全连接的，或者至少是使用邻接矩阵来表示
      Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H
    else:
      # 在稀疏图表示中，只存储图中实际存在的边
      Vh = self.V(h[edge_index[1]])  # E x H

    # 边更新和门控的线性变换 门控：门控机制可以决定哪些信息可以通过边从一个节点传递到另一个节点。
    Ah = self.A(h)  # B x V x H, source 点
    Bh = self.B(h)  # B x V x H, target 点
    Ce = self.C(e)  # B x V x V x H / E x H 边

    # 更新边特征并计算边门控
    if not sparse:
      e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
    else:
      e = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H

    # 根据聚合到的信息更新节点特征
    h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=sparse)  # B x V x H

    # 归一化节点特征
    if not sparse:
      h = self.norm_h(
          h.view(batch_size * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
    else:
      h = self.norm_h(h) if self.norm_h else h

    # 归一化边的特征
    if not sparse:
      e = self.norm_e(
          e.view(batch_size * num_nodes * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e
    else:
      e = self.norm_e(e) if self.norm_e else e

    # Apply non-linearity
    h = F.relu(h)
    e = F.relu(e)

    # Make residual connection
    if mode == "residual":
      h = h_in + h
      e = e_in + e

    return h, e

  def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False):
    """对图神经网络中的节点进行特征聚合

    Args:
        In Dense version:（稠密图）
          Vh: Neighborhood features (B x V x V x H) 邻域特征，四维张量（B x V x V x H）
          graph: Graph adjacency matrices (B x V x V) 图的邻接矩阵
          gates: Edge gates (B x V x V x H) 门控 用于控制特征的聚合 允许模型在聚合邻居节点特征时，对不同边的重要性进行区分，从而使得信息的传播更加有选择性
          mode: str 聚合模式
        In Sparse version:（稀疏图）
          Vh: Neighborhood features (E x H) 稀疏图是二维张量（E x H）
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix) 表示了一个 V x V 的邻接矩阵，
          但只存储了 E 条边的信息。这意味着这个图是稀疏的，大部分位置的权重为零。是一种特殊的表示
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E) 边的索引  用于访问和操作 torch_sparse.SparseTensor 对象
          第一行存储所有边的源节点索引。
          第二行存储所有边的目标节点索引
        sparse: Whether to use sparse tensors (True/False )
    Returns:
        聚合后的邻域特征，形状为 (B x V x H)
    """
    # Perform feature-wise gating mechanism
    Vh = gates * Vh  # B x V x V x H 使用边门控 gates 对特征 Vh 进行加权，以便在聚合过程中考虑边的重要性。

    # Enforce graph structure through masking
    # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

    # Aggregate neighborhood features
    if not sparse:
      # 对于密集图的两种聚合方式
      if (mode or self.aggregation) == "mean": #周围点的均值聚合
        return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
      elif (mode or self.aggregation) == "max": #周围点的最大值聚合
        return torch.max(Vh, dim=2)[0]
      else: #默认执行求和操作
        return torch.sum(Vh, dim=2)
    else: #对于稀疏图的聚合方式
      sparseVh = SparseTensor(
          row=edge_index[0],
          col=edge_index[1],
          value=Vh,
          sparse_sizes=(graph.size(0), graph.size(1))
      )

      if (mode or self.aggregation) == "mean":
        return sparse_mean(sparseVh, dim=1)

      elif (mode or self.aggregation) == "max":
        return sparse_max(sparseVh, dim=1)

      else:
        return sparse_sum(sparseVh, dim=1)


class PositionEmbeddingSine(nn.Module):
  """生成位置编码，通常用于模型中以提供序列中每个元素的位置信息

  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    '''
    num_pos_feats=64,位置编码的特征维度数，默认为 64
    temperature: 控制编码频率的温度参数，默认为 10000
    normalize: 是否对坐标进行归一化处理，默认为 False
    scale: 用于归一化的尺度因子，默认为 2 * math.pi。
    '''
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    '''
    根据城市的坐标计算每个城市的位置编码
    dim_t 是用来生成每个维度的频率温度因子，这些因子将用于后续的正弦和余弦函数计算中，以生成具有不同频率的位置编码
    适用于需要二维空间位置信息的场景，如 TSP 问题或其他需要处理 x 和 y 坐标的图形问题
    输出形状通常为 [batch_size, num_points, 2]
    '''
    # 从输入中提取出对应的坐标信息
    y_embed = x[:, :, 0]
    x_embed = x[:, :, 1]
    if self.normalize:
      # eps = 1e-6
      y_embed = y_embed * self.scale
      x_embed = x_embed * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

    # 使用了正弦和余弦函数来创建每个城市的位置编码
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
    return pos


class ScalarEmbeddingSine(nn.Module):
  '''另一个用于生成位置编码的神经网络模块

 用于生成标量数据的位置编码，适用于一维序列数据，但可以处理多维输入，例如时间序列分析、音频信号处理等。
 输出[batch_size, seq_len, num_pos_feats]
  '''
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return pos_x


class ScalarEmbeddingSine1D(nn.Module):
  '''
  用于生成一维序列的位置编码
 输出 [batch_size, num_pos_feats]（对于一维序列）或 [batch_size, seq_len, num_pos_feats]
  '''
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x


def run_sparse_layer(layer, time_layer, out_layer, adj_matrix, edge_index, add_time_on_edge=True):
  '''根据是否将时间嵌入加在边上的条件，来更新节点和边的特征

  Args:
    layer:要运行的稀疏图层。
    time_layer:一个处理时间嵌入的层
    out_layer:输出层，通常用于转换边特征
    adj_matrix:邻接矩阵，表示图的结构
    edge_index: 边的索引，用于稀疏图的表示
    add_time_on_edge:一个布尔值，指示是否将时间嵌入加在边

  Returns: 将时间步嵌入后的特征
  '''
  def custom_forward(*inputs):
    x_in = inputs[0]
    e_in = inputs[1]
    time_emb = inputs[2]
    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
    if add_time_on_edge:
      e = e + time_layer(time_emb)
    else:
      x = x + time_layer(time_emb)
    x = x_in + x
    e = e_in + out_layer(e)
    return x, e
  return custom_forward

  # 对于非稀疏图的时间步特征嵌入特征
  # def run_dense_layer(layer, time_layer, out_layer, adj_matrix, add_time_on_edge=True):
  #   def custom_forward(x_in, e_in=None, time_emb=None):
  #     # 对于非稀疏图，我们假设 adj_matrix 是一个密集矩阵
  #     # 并且 layer 函数被设计为接受密集输入
  #
  #     # 更新节点特征
  #     x = layer(x_in, adj_matrix=adj_matrix)
  #
  #     # 如果提供了边特征和时间嵌入，则更新边特征
  #     if e_in is not None and time_emb is not None:
  #       if add_time_on_edge:
  #         e = e_in + time_layer(time_emb)  # 将时间嵌入添加到边特征上
  #       else:
  #         # 如果不将时间信息添加到边上，这里可以选择忽略时间嵌入或将其添加到节点特征上
  #         # 例如，我们可以选择将时间嵌入添加到节点特征上
  #         x = x + time_layer(time_emb)
  #     else:
  #       # 如果没有边特征但有时间嵌入，可以选择将时间嵌入添加到节点特征上
  #       if time_emb is not None:
  #         x = x + time_layer(time_emb)
  #
  #     # 如果有边特征，则进一步更新边特征
  #     if e_in is not None:
  #       e = out_layer(e)  # 假设 out_layer 是给定的用于更新边特征的层
  #
  #     # 将更新的特征加到原始特征上
  #     x = x_in + x
  #     if e_in is not None:
  #       e = e_in + e
  #
  #     return x, e
  #
  #   return custom_forward

# 另一种时间步嵌入的方法
# def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
#   """构建正弦嵌入向量（来自 Fairseq）。 也就是处理时间步嵌入
#
#   这与 tensor2tensor 中的实现相匹配，但与 "Attention Is All You Need" 第3.5节中的描述略有不同。
#   Args:
#     timesteps: jnp.ndarray: 在这些时间步生成嵌入向量
#     embedding_dim: int:要生成的嵌入向量的维度
#     dtype: 生成的嵌入向量的数据类型
#
#   Returns:
#    形状为 `(len(timesteps), embedding_dim)` 的嵌入向量
#   """
#   assert len(timesteps.shape) == 1 #确保 timesteps 是一维数组。
#
#   # 这一步的处理因为包含时间步的嵌入的cond在上面也要进行傅里叶特征的提取
#   # 所以要对这里进行一个重缩放
#   timesteps *= 1000. #将时间步缩放1000倍，这可能是为了调整后续正弦和余弦函数的输入范围。
#   half_dim = embedding_dim // 2  #后续一半用作正弦 一半用作余弦
#   emb = np.log(10000) / (half_dim - 1) #用于生成正弦和余弦波形的对数尺度。
#   '''生成正弦和余弦波形
#   在许多序列模型中，特别是基于Transformer的模型，时间步嵌入通常设计为具有频率衰减的特性。
#   这是因为我们希望模型能够捕捉到不同时间步之间的相对位置关系，而较高的频率（即较小的时间间隔）
#   通常对应于这些细微的位置差异。通过让频率随着时间步的增加而递减，模型可以更敏感地响应序列中较短距离的关系。
#   '''
#   emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb) #生成的是一个递减的序列,对应于不同频率的振幅系数
#   emb = timesteps.astype(dtype)[:, None] * emb[None, :]  # 每个时间步 timesteps 与递减的振幅系数序列相乘，从而为每个时间步生成一组特定的频率系数
#   #合并正弦和余弦-生成实际的正余弦波形
#   emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1) #shape (timesteps, embedding_dim)
#
#   # 因为一半用作正弦 一半用作余弦，但是如果dim是奇数，就用0填充
#   # ((0, 0, 0), (0, 1, 0)) 表示在第一个维度（时间步）的开始和结束不填充任何元素，
#   # 在第二个维度（频率系数）的末尾填充一个元素。
#   # 这样做的结果是，如果 embedding_dim 是奇数，就在 emb 的最后一列添加一个零。
#   if embedding_dim % 2 == 1:  # 零填充
#     '''
#     jax.lax.pad(array, padding_value, padding_config)
#     padding_config[[low_pad, high_pad],[low_pad, high_pad]]
#     low_pad 表示在第一维度（行）的开始（顶部）添加的填充元素的数量。
#     high_pad 表示在第一维度（行）的末尾（底部）添加的填充元素的数量。
#     low_pad 表示在第二维度（列）的开始（左侧）添加的填充元素的数量。
#     high_pad 表示在第二维度（列）的末尾（右侧）添加的填充元素的数量。
#     '''
#     emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
#   assert emb.shape == (timesteps.shape[0], embedding_dim)
#   return emb


class GNNEncoder(nn.Module):
  """Configurable GNN Encoder
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               *args, **kwargs):
    '''

    Args:
      n_layers: GNN 层的数量
      hidden_dim:隐藏层的维度
      out_channels: 输出通道数，默认为 1
      aggregation:聚合函数的类型，如 "sum"、"mean"、"max" 等
      norm:归一化类型，如 "batch"、"instance"、"layer" 等
      learn_norm:是否学习归一化参数
      track_norm:是否追踪归一化的统计数据
      gated:是否使用门控机制
      sparse:是否使用稀疏图表示
      use_activation_checkpoint:是否使用激活检查点技术来减少内存使用
      node_feature_only:是否仅使用节点特征而不包括位置或时间嵌入
      *args:
      **kwargs:
    '''
    super(GNNEncoder, self).__init__()
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    # 将节点特征与边的特征引入非线性
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    if not node_feature_only:
      #如果不只是用节点特征就只对节点特征完成位置编码还有对边特征的位置编码
      self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
      self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    else:
      # 如果只是用节点特征就只对节点特征完成位置编码
      self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    ) #创建一个时间序列模型 用于完成后续的时间步嵌入 输出维度是 time_embed_dim
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    ) #创建一个序列模型 out，用于最终的图表示生成

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ]) # 完整的GNN 层列表 这是作为编码器中的一层 但是其中存在许多对数据的处理 层

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ]) # 为每个 GNN 层提供时间嵌入的转换

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ]) # 包含归一化、激活函数和线性层，用于处理每个 GNN 层的输出
    self.use_activation_checkpoint = use_activation_checkpoint # 是否使用激活检查点技术

  def dense_forward(self, x, graph, timesteps, edge_index=None):
    """非稀疏图的前向传播函数 包含了对时间步的嵌入

    Args:
        x: 输入的节点坐标，形状为 (B x V x 2)，其中 B 是批次大小，V 是图中节点的数量，2 表示每个节点有 x 和 y 坐标
        graph: 图的邻接矩阵 (B x V x V)
        timesteps: 输入的节点时间步信息，形状为 (B)
        edge_index:边的索引 (2 x E)
    Returns:
        更新后的边的特征 (B x V x V)
    """
    # Embed edge features
    del edge_index # 因为处理的是非稀疏图 不需要边的索引
    x = self.node_embed(self.pos_embed(x))
    e = self.edge_embed(self.edge_pos_embed(graph))
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    graph = torch.ones_like(graph).long()

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError # 暂时没有使用检查点

      x, e = layer(x, e, graph, mode="direct")
      if not self.node_feature_only:
        e = e + time_layer(time_emb)[:, None, None, :]
      else:
        x = x + time_layer(time_emb)[:, None, :]
      x = x_in + x
      e = e_in + out_layer(e)
    e = self.out(e.permute((0, 3, 1, 2)))
    return e #非稀疏图更新后的边的特征

  def sparse_forward(self, x, graph, timesteps, edge_index):
    """稀疏图更新后的边的特征

    Args:
        x: 输入的节点坐标，形状为 ( V x 2)，其中 V 是图中节点的数量，2 表示每个节点有 x 和 y 坐标
        graph:图的边特征，形状为 (E,)，其中 E 是边的数量
        timesteps: 输入的边时间步特征，形状为 (E,)
        edge_index: 图的邻接矩阵索引，形状为 (2 x E)
    Returns:
        更新后的边的特征 (E x H)
    """
    # Embed edge features
    x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
    e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
    e = self.out(e).reshape(-1, edge_index.shape[1]).permute((1, 0)) #(E, H)
    return e #稀疏图更新后的边的特征

  def sparse_forward_node_feature_only(self, x, timesteps, edge_index):
    '''
    这里没有考虑非稀疏图的只有节点特征的情况就是 在稀疏图中，边的信息通常是稀疏的，因此需要特别处理
    在非稀疏图中，即使只使用节点特征，节点间的交互也可以通过邻接矩阵有效地建模
    Args:
      x: 输入的节点特征
      timesteps: 输入的边时间步特征，形状为 (E,)
      edge_index: 图的邻接矩阵索引，形状为 (2 x E)

    Returns: 更新后的节点的特征 (V x 2)

    '''
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  def sparse_encoding(self, x, e, edge_index, time_emb):
    '''用于更新节点和边特征，输入的节点特征与边特征进行重新编码将时间步嵌入两个特征

    Args:
      x: 节点特征张量 [V,H]
      e:边特征张量，形状为 [E, H]
      edge_index:边的索引信息，形状为 [2, E]
      time_emb:时间嵌入向量，形状为 [1, T] 或 [E, T]，取决于是否为每条边或每个时间步提供时间信息，T 是时间特征的维度

    Returns:更新后的节点特征和边特征，通常为 x 和 e

    '''
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones_like(edge_index[0].float()),
        sparse_sizes=(x.shape[0], x.shape[0]),
    ) #邻接矩阵
    adj_matrix = adj_matrix.to(x.device)

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        single_time_emb = time_emb[:1] #获取当前时间步

        run_sparse_layer_fn = functools.partial(
            run_sparse_layer,
            add_time_on_edge=not self.node_feature_only
        ) #创建新的函数

        out = activation_checkpoint.checkpoint(
            run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
            x_in, e_in, single_time_emb
        )
        x = out[0]
        e = out[1]
      else:
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
        if not self.node_feature_only:
          e = e + time_layer(time_emb)
        else:
          x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
    return x, e #包含了时间步的信息

  def forward(self, x, timesteps, graph=None, edge_index=None):
    if self.node_feature_only:
      if self.sparse:
        return self.sparse_forward_node_feature_only(x, timesteps, edge_index)
      else: # 对于非稀疏图如果只用节点特征就要报错 ，就两种特征必须同时使用
        raise NotImplementedError
    else:
      if self.sparse:
        return self.sparse_forward(x, graph, timesteps, edge_index)
      else:
        return self.dense_forward(x, graph, timesteps, edge_index)
