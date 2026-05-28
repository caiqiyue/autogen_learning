"""
AutoGen RedisCache 配置详解

本文件详细展示 RedisCache 的配置方法、连接参数、适用场景，
以及如何在企业级环境中部署和使用 Redis 缓存。

作者: AI Learning Team
课程: 第14节 AutoGen缓存机制与成本控制
依赖: pip install redis diskcache
"""

import time
from typing import Optional

# ============================================================
# 1. RedisCache 基础配置
# ============================================================

def basic_redis_cache_config():
    """
    RedisCache 基础配置

    必需的参数:
    - host: Redis 服务器地址
    - port: Redis 端口（默认 6379）

    可选参数:
    - db: 数据库编号（默认 0）
    - password: 密码（如需要）
    - prefix: 缓存键前缀（用于区分不同应用）
    - ttl: 缓存过期时间（秒）
    """
    print("=" * 60)
    print("1. RedisCache 基础配置")
    print("=" * 60)

    # 基础连接配置
    redis_host = "localhost"
    redis_port = 6379
    redis_db = 0

    # 可选的缓存配置
    cache_prefix = "autogen:cache:"  # 缓存键前缀，便于识别和管理
    cache_ttl = 3600  # 缓存有效期（秒），1小时

    print(f"连接参数:")
    print(f"  - Host: {redis_host}")
    print(f"  - Port: {redis_port}")
    print(f"  - DB: {redis_db}")
    print(f"  - Prefix: {cache_prefix}")
    print(f"  - TTL: {cache_ttl} 秒")

    # 实际使用示例（需要取消注释）
    # from diskcache import RedisCache
    # redis_cache = RedisCache(
    #     host=redis_host,
    #     port=redis_port,
    #     db=redis_db,
    #     prefix=cache_prefix,
    #     ttl=cache_ttl
    # )

    print("\n使用示例:")
    print("  from diskcache import RedisCache")
    print("  redis_cache = RedisCache(")
    print("      host='localhost',")
    print("      port=6379,")
    print("      db=0,")
    print("      prefix='autogen:cache:',")
    print("      ttl=3600")
    print("  )")
    print()


# ============================================================
# 2. 带密码的 Redis 连接配置
# ============================================================

def redis_with_auth():
    """
    配置带密码认证的 Redis 连接

    适用场景:
    - 生产环境中需要认证的 Redis 服务
    - 云服务商提供的 Redis 实例（如阿里云、AWS）
    - 企业内网需要安全访问的 Redis
    """
    print("=" * 60)
    print("2. 带密码认证的 Redis 配置")
    print("=" * 60)

    redis_host = "redis.example.com"
    redis_port = 6379
    redis_password = "your-redis-password"  # 请替换为实际密码
    redis_db = 0

    print(f"连接参数:")
    print(f"  - Host: {redis_host}")
    print(f"  - Port: {redis_port}")
    print(f"  - Password: **** (已隐藏)")
    print(f"  - DB: {redis_db}")

    print("\n使用示例:")
    print("  from diskcache import RedisCache")
    print("  redis_cache = RedisCache(")
    print("      host='redis.example.com',")
    print("      port=6379,")
    print("      password='your-password',")
    print("      db=0")
    print("  )")
    print()


# ============================================================
# 3. Redis Sentinel 高可用配置
# ============================================================

def redis_sentinel_config():
    """
    Redis Sentinel 高可用配置

    适用场景:
    - 生产环境需要高可用性
    - 主从自动切换
    - 故障转移自动处理

    工作原理:
    - Sentinel 监控主从节点状态
    - 主节点故障时自动选举新主
    - 客户端自动连接新主节点
    """
    print("=" * 60)
    print("3. Redis Sentinel 高可用配置")
    print("=" * 60)

    # Sentinel 配置
    sentinels = [
        ("sentinel1.example.com", 26379),
        ("sentinel2.example.com", 26379),
        ("sentinel3.example.com", 26379)
    ]
    sentinel_service_name = "mymaster"  # 主服务名称
    sentinel_password = "redis-password"  # 主节点密码

    print(f"Sentinel 节点:")
    for host, port in sentinels:
        print(f"  - {host}:{port}")

    print(f"\n服务名称: {sentinel_service_name}")
    print(f"主节点密码: **** (已隐藏)")

    print("\n说明:")
    print("  - Sentinel 会监控主从节点健康状态")
    print("  - 主节点故障时自动进行故障转移")
    print("  - 客户端通过 Sentinel 获取当前主节点")
    print()


# ============================================================
# 4. Redis Cluster 集群配置
# ============================================================

def redis_cluster_config():
    """
    Redis Cluster 集群配置

    适用场景:
    - 超大规模缓存需求
    - 需要数据分片存储
    - 高并发读写场景

    分片策略:
    - 自动将数据分布到多个节点
    - 每个节点存储不同的数据片段
    - 客户端自动路由到正确节点
    """
    print("=" * 60)
    print("4. Redis Cluster 集群配置")
    print("=" * 60)

    # Cluster 节点配置
    cluster_nodes = [
        ("node1.example.com", 6379),
        ("node2.example.com", 6379),
        ("node3.example.com", 6379),
        ("node4.example.com", 6379),
        ("node5.example.com", 6379),
        ("node6.example.com", 6379)
    ]

    print("集群节点:")
    for i, (host, port) in enumerate(cluster_nodes):
        role = "主节点" if i < 3 else "从节点"
        print(f"  - {host}:{port} ({role})")

    print("\n说明:")
    print("  - Redis Cluster 使用 hash slot 进行数据分片")
    print("  - 共 16384 个 slot，分布到各主节点")
    print("  - 每个主节点可有多个从节点提供副本")
    print("  - 自动处理节点故障和恢复")
    print()


# ============================================================
# 5. 连接池配置与性能调优
# ============================================================

def connection_pool_config():
    """
    Redis 连接池配置与性能调优

    连接池参数:
    - max_connections: 最大连接数
    - socket_timeout: socket 超时时间
    - socket_connect_timeout: 连接超时时间
    - retry_on_timeout: 超时重试配置

    性能建议:
    - 连接数不宜过大，避免资源浪费
    - 超时时间要合理，平衡等待和失败
    - 启用连接复用，减少连接开销
    """
    print("=" * 60)
    print("5. 连接池配置与性能调优")
    print("=" * 60)

    # 连接池配置参数
    max_connections = 50  # 最大连接数
    socket_timeout = 30  # socket 超时（秒）
    socket_connect_timeout = 5  # 连接超时（秒）
    retry_on_timeout = True  # 超时重试

    print("连接池参数:")
    print(f"  - max_connections: {max_connections}")
    print(f"  - socket_timeout: {socket_timeout}s")
    print(f"  - socket_connect_timeout: {socket_connect_timeout}s")
    print(f"  - retry_on_timeout: {retry_on_timeout}")

    print("\n性能调优建议:")
    print("  1. 根据并发量设置 max_connections")
    print("     - 小型应用: 10-20 连接")
    print("     - 中型应用: 20-50 连接")
    print("     - 大型应用: 50-100+ 连接")
    print("  2. 超时时间设置")
    print("     - 短超时: 适合快速失败的场景")
    print("     - 长超时: 适合需要等待的场景")
    print("  3. 连接复用")
    print("     - 保持长连接，减少连接建立开销")
    print("     - 合理设置连接池大小")
    print()


# ============================================================
# 6. AutoGen 与 RedisCache 集成示例
# ============================================================

def autogen_redis_integration():
    """
    AutoGen 中使用 RedisCache 的完整集成示例

    完整流程:
    1. 创建 RedisCache 实例
    2. 配置到 AutoGen LLM 配置中
    3. 创建 agent 使用该配置
    4. 享受缓存加速
    """
    print("=" * 60)
    print("6. AutoGen 与 RedisCache 集成示例")
    print("=" * 60)

    # 步骤1: 创建 RedisCache 实例
    print("\n步骤1: 创建 RedisCache 实例")
    print("  from diskcache import RedisCache")
    print("  redis_cache = RedisCache(")
    print("      host='localhost',")
    print("      port=6379,")
    print("      db=0,")
    print("      prefix='autogen:',")
    print("      ttl=3600")
    print("  )")

    # 步骤2: 配置到 AutoGen
    print("\n步骤2: 配置到 AutoGen LLM")
    print("  config_list = [")
    print("      {")
    print("          'model': 'gpt-4o',")
    print("          'api_key': 'your-api-key',")
    print("          'cache_seed': redis_cache  # 关键配置")
    print("      }")
    print("  ]")

    # 步骤3: 创建 agent
    print("\n步骤3: 创建 Agent 使用该配置")
    print("  from autogen import ConversableAgent")
    print("  agent = ConversableAgent(")
    print("      'assistant',")
    print("      llm_config={'config_list': config_list}")
    print("  )")

    # 步骤4: 性能监控
    print("\n步骤4: 性能监控示例")
    print("  首次请求: 500-2000ms (实际 API 调用)")
    print("  缓存命中: < 1ms (直接返回)")
    print("  缓存命中率: 可通过监控指标查看")
    print()


# ============================================================
# 7. 适用场景分析与选择建议
# ============================================================

def use_case_analysis():
    """
    各种 Redis 配置的适用场景分析与选择建议

    场景1: 个人开发/小项目
    -> 推荐: 单机 Redis + DiskCache
    -> 原因: 部署简单，成本低

    场景2: 中小型企业应用
    -> 推荐: 带密码的单机 Redis
    -> 原因: 安全性好，配置简单

    场景3: 大型企业/高可用要求
    -> 推荐: Redis Sentinel
    -> 原因: 自动故障转移，高可用

    场景4: 超大规模/高并发
    -> 推荐: Redis Cluster
    -> 原因: 水平扩展，高并发
    """
    print("=" * 60)
    print("7. 适用场景分析与选择建议")
    print("=" * 60)

    scenarios = [
        {
            "name": "个人开发/小项目",
            "recommendation": "单机 Redis 或 DiskCache",
            "reason": "部署简单，成本低，适合学习和原型开发"
        },
        {
            "name": "中小型企业应用",
            "recommendation": "带密码的单机 Redis",
            "reason": "安全性好，配置简单，维护成本低"
        },
        {
            "name": "大型企业/高可用要求",
            "recommendation": "Redis Sentinel",
            "reason": "自动故障转移，保证服务可用性"
        },
        {
            "name": "超大规模/高并发场景",
            "recommendation": "Redis Cluster",
            "reason": "水平扩展，数据分片，支持超高并发"
        }
    ]

    print("\n场景选择指南:")
    print("-" * 70)
    for scenario in scenarios:
        print(f"\n【{scenario['name']}】")
        print(f"  推荐: {scenario['recommendation']}")
        print(f"  原因: {scenario['reason']}")
    print("-" * 70)
    print()


# ============================================================
# 主函数入口
# ============================================================

if __name__ == "__main__":
    """
    主函数 - 运行所有演示
    """
    print("\n" + "=" * 60)
    print("AutoGen RedisCache 配置详解")
    print("=" * 60)
    print()

    # 运行各项演示
    basic_redis_cache_config()
    redis_with_auth()
    redis_sentinel_config()
    redis_cluster_config()
    connection_pool_config()
    autogen_redis_integration()
    use_case_analysis()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n前置要求:")
    print("  1. 安装 Redis: https://redis.io/download")
    print("  2. 安装依赖: pip install redis diskcache")
    print("  3. 确保 Redis 服务正在运行")
    print("\n快速启动 Redis:")
    print("  docker run -d -p 6379:6379 redis:latest")
    print("\n参考资料:")
    print("  - AutoGen 缓存文档: https://microsoft.github.io/autogen/")
    print("  - diskcache 文档: https://grantjenks.com/docs/diskcache/")
    print("  - Redis 文档: https://redis.io/docs/")