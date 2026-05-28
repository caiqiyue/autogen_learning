"""
AutoGen 缓存机制与成本控制 - cache_seed 配置演示

本文件展示 AutoGen 中 cache_seed 参数的各种配置方式及其效果。
cache_seed 用于控制 LLM 响应的缓存，以减少 API 调用次数并降低成本。

作者: AI Learning Team
课程: 第14节 AutoGen缓存机制与成本控制
"""

from typing import Optional, Union
from diskcache import Cache
import time

# ============================================================
# 1. 基础配置 - cache_seed=None (禁用缓存)
# ============================================================

def demo_cache_disabled():
    """
    演示 cache_seed=None 时禁用缓存的原理

    原理说明:
    - 当 cache_seed 设置为 None 时，AutoGen 不会对任何 LLM 响应进行缓存
    - 每次请求都会直接发送到 LLM API，产生真实的 API 调用费用
    - 适用于需要每次都获取最新结果的场景（如实时数据查询）

    使用场景:
    - 调试模式：方便调试日志追踪每次请求
    - 实时性要求高的应用：如股票价格、新闻资讯等
    - 开发测试阶段：避免缓存干扰测试结果
    """
    print("=" * 60)
    print("1. cache_seed=None 禁用缓存演示")
    print("=" * 60)

    # 配置示例 - 禁用缓存
    config_list = [
        {
            "model": "gpt-4o",
            "api_key": "your-api-key",  # 请替换为实际 API Key
            "cache_seed": None  # 关键配置：None 表示禁用缓存
        }
    ]

    print("配置说明:")
    print("  - cache_seed: None")
    print("  - 效果: 每次请求都调用 LLM API，不使用缓存")
    print("  - 优点: 总是获取最新结果")
    print("  - 缺点: API 调用次数多，成本较高")
    print()


# ============================================================
# 2. DiskCache 本地磁盘缓存配置
# ============================================================

def demo_disk_cache():
    """
    演示 DiskCache 本地磁盘缓存的配置与适用场景

    原理说明:
    - DiskCache 将 LLM 响应存储在本地磁盘上
    - 缓存 key 由请求的 prompt + model + temperature 等参数生成 hash
    - 相同参数的请求会命中磁盘缓存，直接返回历史响应

    适用场景:
    - 个人开发或小团队使用
    - 缓存数据不需要跨机器共享
    - 数据量适中（磁盘空间有限）
    - 对响应速度要求不高，但希望节省成本

    性能特点:
    - 首次请求：正常 API 调用延迟
    - 缓存命中：几乎没有延迟（毫秒级）
    - 磁盘 I/O 是主要瓶颈
    """
    print("=" * 60)
    print("2. DiskCache 本地磁盘缓存配置演示")
    print("=" * 60)

    cache_dir = "./cache/disk_cache"

    # 配置 DiskCache
    # DiskCache 是 diskcache 库提供的类，需要先安装: pip install diskcache
    disk_cache = Cache(cache_dir)

    config_list = [
        {
            "model": "gpt-4o",
            "api_key": "your-api-key",
            "cache_seed": disk_cache  # 使用 DiskCache 实例作为缓存后端
        }
    ]

    print("配置说明:")
    print("  - cache_seed: Cache('./cache/disk_cache')")
    print("  - 效果: 使用本地磁盘存储缓存")
    print("  - 优点:")
    print("    * 缓存持久化，重启后仍可用")
    print("    * 无需额外服务，部署简单")
    print("    * 适合个人/小团队场景")
    print("  - 缺点:")
    print("    * 缓存数据存在本地，不能跨机器共享")
    print("    * 磁盘空间有限，不适合大规模缓存")
    print("    * 磁盘 I/O 性能低于内存")
    print()

    # 演示缓存命中测试
    print("缓存命中测试:")
    start_time = time.time()

    # 第一次请求（模拟）
    print("  第一次请求: 直接调用 API...")

    # 第二次请求（模拟）
    print("  第二次请求: 命中缓存，响应时间 < 1ms")

    elapsed = time.time() - start_time
    print(f"  总耗时: {elapsed:.4f} 秒")
    print()


# ============================================================
# 3. RedisCache 分布式缓存配置
# ============================================================

def demo_redis_cache_basic():
    """
    演示 RedisCache 的基础配置

    前提条件:
    - 需要安装 Redis 服务器
    - 需要安装 redis python 包: pip install redis
    - 需要安装 diskcache 包: pip install diskcache

    适用场景:
    - 企业级应用，需要跨服务共享缓存
    - 多台服务器部署，需要一致的缓存体验
    - 高并发场景，需要高性能缓存
    - 分布式部署环境

    性能特点:
    - Redis 是内存数据库，读取速度极快
    - 支持网络访问，跨机器共享
    - 支持集群部署，高可用
    """
    print("=" * 60)
    print("3. RedisCache 基础配置演示")
    print("=" * 60)

    # Redis 配置参数
    redis_host = "localhost"
    redis_port = 6379
    redis_db = 0
    redis_password = None  # 如果有密码请设置

    # 使用 AutoGen 提供的 RedisCache
    # from autogen.cache.decorators import RedisCache
    # redis_cache = RedisCache(redis_host, redis_port, redis_db, redis_password)

    config_list = [
        {
            "model": "gpt-4o",
            "api_key": "your-api-key",
            # "cache_seed": redis_cache  # 实际使用时请取消注释
        }
    ]

    print("配置说明:")
    print(f"  - Redis 地址: {redis_host}:{redis_port}")
    print("  - cache_seed: RedisCache 实例")
    print("  - 效果: 使用 Redis 作为缓存后端")
    print("  - 优点:")
    print("    * 内存级访问，超低延迟")
    print("    * 跨服务共享缓存")
    print("    * 支持集群，高可用")
    print("    * 适合生产环境")
    print("  - 缺点:")
    print("    * 需要额外部署 Redis 服务")
    print("    * 网络延迟（尽管很小）")
    print("    * 增加系统复杂度")
    print()


# ============================================================
# 4. 缓存对响应速度的影响分析
# ============================================================

def analyze_cache_speed_impact():
    """
    分析缓存对响应速度的影响

    对比场景:
    1. 无缓存 (cache_seed=None)
    2. DiskCache (本地磁盘)
    3. RedisCache (内存数据库)

    典型延迟对比:
    - API 调用延迟: 500ms - 3000ms（取决于模型和网络）
    - DiskCache 命中: 1ms - 10ms
    - RedisCache 命中: 0.1ms - 1ms
    """
    print("=" * 60)
    print("4. 缓存对响应速度的影响分析")
    print("=" * 60)

    print("\n延迟对比表:")
    print("-" * 50)
    print(f"{'缓存类型':<20} {'首次请求':<15} {'缓存命中':<15}")
    print("-" * 50)
    print(f"{'无缓存(None)':<20} {'500-3000ms':<15} {'N/A':<15}")
    print(f"{'DiskCache':<20} {'500-3000ms':<15} {'1-10ms':<15}")
    print(f"{'RedisCache':<20} {'500-3000ms':<15} {'0.1-1ms':<15}")
    print("-" * 50)

    print("\n速度提升倍数:")
    print("  - DiskCache vs 无缓存: 50-3000x")
    print("  - RedisCache vs 无缓存: 500-30000x")
    print()


# ============================================================
# 5. 企业成本优化策略
# ============================================================

def enterprise_cost_optimization():
    """
    企业级成本优化策略

    策略1: 智能缓存分级
    - 热数据: 使用 RedisCache，毫秒级响应
    - 温数据: 使用 DiskCache，低成本持久化
    - 冷数据: 及时清理，避免浪费空间

    策略2: 缓存有效期管理
    - 短期缓存: 适合频繁变化的查询（如用户状态）
    - 长期缓存: 适合稳定的知识问答（如产品说明）

    策略3: 缓存键设计优化
    - 合理设计缓存 key，减少不必要的重复查询
    - 排除不相关的参数（如调试标志）

    策略4: 成本监控与告警
    - 监控 API 调用次数和缓存命中率
    - 设置成本阈值告警

    策略5: 混合缓存策略
    - 本地缓存 + 分布式缓存结合
    - 减少网络开销，提高响应速度
    """
    print("=" * 60)
    print("5. 企业成本优化策略")
    print("=" * 60)

    print("\n策略1: 智能缓存分级")
    print("  热数据 -> RedisCache (内存，高速)")
    print("  温数据 -> DiskCache (磁盘，持久)")
    print("  冷数据 -> 定期清理 (节省空间)")

    print("\n策略2: 缓存有效期管理")
    print("  短期缓存 (5-30分钟): 适合频繁变化的查询")
    print("  长期缓存 (1-24小时): 适合稳定的知识问答")
    print("  永不过期: 仅用于不变的数据")

    print("\n策略3: 缓存键设计优化")
    print("  - 包含所有影响结果的参数")
    print("  - 排除不影响结果的参数")
    print("  - 使用哈希压缩长 prompt")

    print("\n策略4: 成本监控与告警")
    print("  - 记录 API 调用次数")
    print("  - 计算缓存命中率")
    print("  - 设置月度成本阈值")

    print("\n策略5: 混合缓存策略示例")
    print("  local_cache (DiskCache) + shared_cache (RedisCache)")
    print("  首次查询: API -> Redis -> 本地")
    print("  后续查询: 本地直接返回")
    print()


# ============================================================
# 主函数入口
# ============================================================

if __name__ == "__main__":
    """
    主函数 - 运行所有演示
    """
    print("\n" + "=" * 60)
    print("AutoGen 缓存机制与成本控制 - 配置演示")
    print("=" * 60)
    print()

    # 运行各项演示
    demo_cache_disabled()
    demo_disk_cache()
    demo_redis_cache_basic()
    analyze_cache_speed_impact()
    enterprise_cost_optimization()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n参考资料:")
    print("  - AutoGen 官方文档: https://microsoft.github.io/autogen/")
    print("  - diskcache 库: https://grantjenks.com/docs/diskcache/")
    print("  - Redis 官网: https://redis.io/")