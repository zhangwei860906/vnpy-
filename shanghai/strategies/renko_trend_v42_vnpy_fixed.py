# -*- coding: utf-8 -*-
"""
文件名: renko_trend_v42_vnpy_fixed.py
策略名: RenkoTrendV42VnPy

V4.2 Renko 趋势策略 - VnPy适配修复整合版
========================================

修复说明:
本版本整合了之前讨论的所有A/B/C类修复，包括：
1. A01: reduce_on_pullback_bricks参数修复
2. A02: trend_has_advanced_since_addon触发阈值修复
3. A03+A05: 仓位与均价计算的稳健性修复
4. A04: 订单超时状态机竞态修复
5. A06: 启动期保护机制
6. B01: Regime切换识别滞后优化
7. B03: 批量砖生成处理优化
8. B05: Hard reversal冷却机制完善
9. B07: Shock识别优化
10. C01: 价格归一化方向处理
11. C02: 参数校验增强
12. C07: 重复扫描修复
13. C09: 日志优化
14. 历史数据加载初始化
15. 资金管理模型增强

所有修复都添加了独立的enable_xxx_fix开关，便于测试和回退。
"""

from __future__ import annotations

import math
import time
import statistics
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Dict, Any, Tuple, Deque
from datetime import datetime, timedelta

from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
)
from vnpy.trader.constant import (
    Direction,
    Offset,
    Interval,
    OrderType,
    Status,
)


# ============================================================
# 常量定义
# ============================================================
REGIME_TREND_CLEAN = "TREND_CLEAN"
REGIME_TREND_NOISY = "TREND_NOISY"
REGIME_MEAN_REVERT = "MEAN_REVERT"
REGIME_SHOCK = "SHOCK_DISLOCATED"

POSITION_FLAT = "FLAT"
POSITION_LONG = "LONG"
POSITION_SHORT = "SHORT"

ROLE_NONE = "NONE"
ROLE_PROBE = "PROBE"
ROLE_CORE = "CORE"

INTENT_OPEN_LONG = "OPEN_LONG"
INTENT_OPEN_SHORT = "OPEN_SHORT"
INTENT_CLOSE_LONG = "CLOSE_LONG"
INTENT_CLOSE_SHORT = "CLOSE_SHORT"


# ============================================================
# 数据结构定义
# ============================================================
@dataclass
class RenkoBrick:
    brick_id: int
    ts: int
    open_price: float
    close_price: float
    direction: int
    source_price: float
    source_tick_ts: int
    bricks_generated_in_tick: int
    run_id: int


@dataclass
class StructureFeatures:
    directional_persistence: float = 0.0
    reversal_density: float = 0.0
    avg_run_length: float = 0.0
    max_run_length: float = 0.0
    net_displacement_efficiency: float = 0.0
    pullback_share: float = 0.0
    extension_score: float = 0.0
    chop_score: float = 0.0
    current_run_length: int = 0
    up_count_window: int = 0
    down_count_window: int = 0


@dataclass
class DecisionContext:
    ts: int
    position_state: str
    position_role: str
    net_position_qty: float
    avg_entry_price: float
    unrealized_pnl: float
    unrealized_pnl_bricks: float
    addon_stage: int
    addon_total_qty: float
    entry_anchor_price: float
    last_addon_anchor_price: float
    reduce_anchor_price: float
    regime: str
    tempo_state: str
    loss_pause_active: bool
    hard_reversal_ready: bool


@dataclass
class TradeBatch:
    batch_id: str
    direction: str
    entry_ts: int
    entry_price: float
    entry_count: int = 0
    exit_count: int = 0
    realized_pnl: float = 0.0
    realized_pnl_bricks: float = 0.0
    holding_bricks: int = 0
    mfe: float = 0.0
    mae: float = 0.0
    regime_history: List[str] = field(default_factory=list)
    total_entry_qty: float = 0.0
    total_exit_qty: float = 0.0


# ============================================================
# 内部持仓跟踪器（A03+A05修复）
# ============================================================
class PositionLedger:
    """独立的持仓与成本跟踪器，不依赖外部时序 [FIX: A03+A05]"""
    
    def __init__(self, contract_size: float = 1.0):
        self.contract_size = contract_size
        self.long_qty = 0.0
        self.short_qty = 0.0
        self.long_cost = 0.0  # 总成本 = 持仓量 * 均价
        self.short_cost = 0.0
        self.entry_trades: List[Tuple[float, float]] = []  # (价格, 数量) 记录所有开仓
        
    def apply_trade(self, direction: str, offset: str, price: float, volume: float) -> None:
        """应用成交"""
        if offset == "OPEN":
            if direction == "LONG":
                self.long_qty += volume
                self.long_cost += price * volume
                self.entry_trades.append((price, volume))
            else:  # SHORT
                self.short_qty += volume
                self.short_cost += price * volume
                # 空头持仓用负数表示在entry_trades中
                self.entry_trades.append((price, -volume))
        else:  # CLOSE
            if direction == "LONG":  # 平多 -> 减少空头
                self.short_qty = max(0, self.short_qty - volume)
                if self.short_qty <= 1e-12:
                    self.short_cost = 0.0
            else:  # SHORT -> 平空 -> 减少多头
                self.long_qty = max(0, self.long_qty - volume)
                if self.long_qty <= 1e-12:
                    self.long_cost = 0.0
    
    @property
    def net_qty(self) -> float:
        """净持仓量（多头为正）"""
        return self.long_qty - self.short_qty
    
    @property
    def avg_entry_price(self) -> float:
        """基于实际持仓的加权平均成本"""
        if self.net_qty > 0:  # 净多头
            return self.long_cost / self.long_qty if self.long_qty > 0 else 0.0
        elif self.net_qty < 0:  # 净空头
            return self.short_cost / self.short_qty if self.short_qty > 0 else 0.0
        return 0.0
    
    def reset(self) -> None:
        """重置所有状态（仓位归零时调用）"""
        self.long_qty = 0.0
        self.short_qty = 0.0
        self.long_cost = 0.0
        self.short_cost = 0.0
        self.entry_trades.clear()


# ============================================================
# 主策略类
# ============================================================
class RenkoTrendV42VnPy(CtaTemplate):
    """V4.2 Renko趋势策略 - VnPy适配修复整合版"""
    
    author = "Strategy Designer"

    # ==================== 一、基本与合约参数 ====================
    symbol = "BTCUSDT"
    inst_type = "SWAP"
    margin_coin = "USDT"
    position_mode = "net"
    contract_size = 1.0
    default_leverage = 1
    min_qty = 0.001
    qty_step = 0.001
    price_tick = 0.1

    # ==================== 二、Renko 砖块参数 ====================
    brick_size = 2
    max_bricks_per_bar = 20

    # ==================== 三、核心开平仓触发参数 ====================
    open_trigger_bricks = 2
    close_trigger_bricks = 2
    core_upgrade_profit_bricks = 1

    # ==================== 四、Addon 加仓与 Reduce 减仓参数 ====================
    enable_addon = True
    addon_displacement_bricks = 5
    addon1_trigger_bricks = 3
    addon2_trigger_bricks = 2
    addon3_trigger_bricks = 2
    addon1_ratio = 0.8
    addon2_ratio = 0.6
    addon3_ratio = 0.4
    reduce_on_pullback_bricks = 1
    reduce_addon_ratio = 0.5
    reset_addon_stage_after_reduce = True
    re_addon_after_reduce_trigger_bricks = 3

    # ==================== 五、市场状态识别参数 - 结构特征 ====================
    enable_regime_filter = True
    regime_lookback_bricks = 24
    min_regime_hold_bricks = 3
    persistence_trend_clean_min = 0.65
    reversal_trend_clean_max = 0.35
    efficiency_trend_clean_min = 0.35
    pullback_trend_clean_max = 0.35
    persistence_trend_noisy_min = 0.55
    efficiency_trend_noisy_min = 0.20
    avg_run_trend_noisy_min = 1.8
    trend_clean_enter_threshold = 0.72
    trend_clean_exit_threshold = 0.62
    shock_override_hold = True

    # ==================== 六、市场状态识别参数 - Hurst指数 ====================
    enable_hurst = True
    hurst_lookback_bricks = 64
    hurst_trend_threshold = 0.58
    hurst_mean_revert_threshold = 0.48
    hurst_addon_floor = 0.50

    # ==================== 七、市场状态识别参数 - Tempo与极端状态 ====================
    tempo_window_seconds = 60
    tempo_slow_max = 1
    tempo_normal_max = 3
    tempo_fast_max = 6
    tempo_burst_min = 7
    dislocated_single_bar_bricks = 3
    shock_extension_threshold = 2.5
    shock_score_threshold = 2

    # ==================== 八、仓位与保证金参数 ====================
    margin_mode = "fixed"
    base_margin_fixed = 100.0
    base_margin_ratio = 0.05
    open_ratio_trend_clean = 1.0
    open_ratio_trend_noisy = 0.5
    open_ratio_mean_revert = 0.25
    open_ratio_shock = 0.15

    # ==================== 九、市场状态 -> 最大加仓层数 ====================
    max_addons_trend_clean = 3
    max_addons_trend_noisy = 1
    max_addons_mean_revert = 0
    max_addons_shock = 0

    # ==================== 十、Hard Reversal 硬反转参数 ====================
    enable_hard_reversal = True
    hard_reversal_requires_extreme = True
    hard_reversal_cooldown_bricks = 3

    # ==================== 十一、连续亏损暂停参数 ====================
    enable_loss_pause = True
    max_consecutive_losses = 3
    pause_recover_bricks = 10

    # ==================== 十二、智能滑点控制参数 ====================
    base_slippage_bricks = 0.2
    slippage_multiplier_open = 1.5
    slippage_multiplier_addon = 0.8
    slippage_multiplier_reduce = 1.2
    slippage_multiplier_exit = 2.0
    slippage_multiplier_trend_clean = 0.7
    slippage_multiplier_trend_noisy = 1.0
    slippage_multiplier_mean_revert = 1.2
    slippage_multiplier_shock = 3.0
    limit_order_timeout_ms = 500
    enable_market_fallback = True
    fallback_to_market_after_timeout = True
    cancel_if_slippage_exceeds = True
    max_slippage_for_limit = 0.5

    # ==================== 十三、日志与调试参数 ====================
    log_tick_events = False
    log_brick_events = True
    log_structure_snapshots = True
    log_regime_snapshots = True
    log_rule_evaluation = False
    log_decision_actions = True
    log_order_events = True
    log_position_events = True
    log_risk_events = True
    snapshot_log_every_n_bricks = 5  # [FIX: C09] 默认从1改为5，减少日志频率

    # ==================== 十四、修复开关参数 ====================
    # 所有修复都有独立开关，便于测试和回退
    enable_reduce_logic_fix = True           # A01+A02修复：减仓逻辑优化
    enable_ledger_fix = True                 # A03+A05修复：内部持仓跟踪器
    enable_order_timeout_fix = True          # A04修复：订单超时状态机
    enable_warmup_protection = True          # A06修复：启动期保护
    enable_regime_lag_fix = True             # B01修复：Regime识别滞后优化
    enable_bulk_brick_fix = True             # B03修复：批量砖生成处理
    enable_hard_reversal_fix = True          # B05修复：Hard reversal冷却完善
    enable_shock_detection_fix = True        # B07修复：Shock识别优化
    enable_price_normalization_fix = True    # C01修复：价格归一化方向处理
    enable_param_validation_fix = True       # C02修复：参数校验增强
    enable_duplicate_scan_fix = True         # C07修复：重复扫描修复
    enable_history_loading = True            # 历史数据加载初始化
    enable_risk_based_sizing = True          # 资金管理模型增强
    
    # ==================== 十五、修复相关新增参数 ====================
    # A02修复：减仓所需的最小有利位移
    min_advance_for_reduce_bricks = 2.0
    
    # A06修复：启动期保护参数
    min_bricks_for_trading = 100
    warmup_bypass_regime = True
    
    # B01修复：Regime切换双阈值参数
    trend_clean_enter_hold_bricks = 1
    trend_clean_exit_hold_bricks = 5
    
    # B03修复：批量砖生成处理参数
    bulk_brick_action_cooldown = 1
    
    # B05修复：Hard reversal冷却行为参数
    hard_reversal_full_cooldown = True  # True:冷却期禁止所有开仓, False:仅禁止同向开仓
    
    # B07修复：Shock识别优化参数
    shock_enter_confirm_bricks = 2
    shock_exit_confirm_bricks = 3
    
    # 资金管理增强参数
    risk_ratio = 0.02           # 单笔交易最大风险占账户总资金比例
    account_balance = 10000.0   # 账户初始权益
    contract_face_value = 0.001 # 单张合约对应的标的数量
    stop_loss_bricks = 2        # 固定止损砖块数
    
    # 减仓冷却参数
    reduce_cooldown_bricks = 3

    # 参数列表更新
    parameters = [
        "brick_size", "max_bricks_per_bar",
        "open_trigger_bricks", "close_trigger_bricks", "core_upgrade_profit_bricks",
        "enable_addon", "addon_displacement_bricks",
        "addon1_trigger_bricks", "addon2_trigger_bricks", "addon3_trigger_bricks",
        "addon1_ratio", "addon2_ratio", "addon3_ratio",
        "reduce_addon_ratio", "re_addon_after_reduce_trigger_bricks",
        "enable_regime_filter", "regime_lookback_bricks", "min_regime_hold_bricks",
        "persistence_trend_clean_min", "reversal_trend_clean_max",
        "efficiency_trend_clean_min", "pullback_trend_clean_max",
        "persistence_trend_noisy_min", "efficiency_trend_noisy_min", "avg_run_trend_noisy_min",
        "trend_clean_enter_threshold", "trend_clean_exit_threshold",
        "enable_hurst", "hurst_lookback_bricks",
        "hurst_trend_threshold", "hurst_mean_revert_threshold", "hurst_addon_floor",
        "dislocated_single_bar_bricks", "shock_extension_threshold", "shock_score_threshold",
        "margin_mode", "base_margin_fixed", "base_margin_ratio",
        "open_ratio_trend_clean", "open_ratio_trend_noisy", "open_ratio_mean_revert", "open_ratio_shock",
        "max_addons_trend_clean", "max_addons_trend_noisy", "max_addons_mean_revert", "max_addons_shock",
        "enable_hard_reversal", "hard_reversal_cooldown_bricks",
        "enable_loss_pause", "max_consecutive_losses", "pause_recover_bricks",
        "base_slippage_bricks",
        "slippage_multiplier_open", "slippage_multiplier_addon",
        "slippage_multiplier_reduce", "slippage_multiplier_exit",
        "slippage_multiplier_trend_clean", "slippage_multiplier_trend_noisy",
        "slippage_multiplier_mean_revert", "slippage_multiplier_shock",
        "limit_order_timeout_ms", "enable_market_fallback",
        "fallback_to_market_after_timeout", "max_slippage_for_limit",
        "snapshot_log_every_n_bricks",
        # 修复开关参数
        "enable_reduce_logic_fix", "enable_ledger_fix", "enable_order_timeout_fix",
        "enable_warmup_protection", "enable_regime_lag_fix", "enable_bulk_brick_fix",
        "enable_hard_reversal_fix", "enable_shock_detection_fix", "enable_price_normalization_fix",
        "enable_param_validation_fix", "enable_duplicate_scan_fix", "enable_history_loading",
        "enable_risk_based_sizing",
        # 修复相关新增参数
        "min_advance_for_reduce_bricks", "min_bricks_for_trading", "warmup_bypass_regime",
        "trend_clean_enter_hold_bricks", "trend_clean_exit_hold_bricks",
        "bulk_brick_action_cooldown", "hard_reversal_full_cooldown",
        "shock_enter_confirm_bricks", "shock_exit_confirm_bricks",
        "risk_ratio", "account_balance", "contract_face_value", "stop_loss_bricks",
        "reduce_cooldown_bricks",
    ]

    # 变量列表更新
    variables = [
        "current_regime",
        "candidate_regime",
        "position_role",
        "addon_stage",
        "consecutive_loss_batches",
        "loss_pause_active",
        "regime_hold_bricks",
        "hard_reversal_cooldown_remaining",
        "total_new_bricks",
        "avg_entry_price",
        "unrealized_pnl",
        "unrealized_pnl_bricks",
        "warmup_complete",  # A06修复
        "reduce_cooldown_remaining",  # A01修复
        "last_bulk_brick_action_id",  # B03修复
        "shock_enter_count",  # B07修复
        "shock_exit_count",  # B07修复
    ]

    # ============================================================
    # 内部类：智能滑点与订单控制器
    # ============================================================
    class SlippageController:
        def __init__(self, parent_strategy: "RenkoTrendV42VnPy"):
            self.strategy = parent_strategy
            self.active_orders: Dict[str, Dict[str, Any]] = {}
            self.order_metas: Dict[str, Dict[str, Any]] = {}
            self.pending_market_fallback: List[Tuple[str, str, float, str]] = []
            
            # [FIX: A04] 新增：已标记超时的订单集合
            self._timeout_flagged_orders: Set[str] = set()

        def calculate_dynamic_slippage(self, action: str, regime: str) -> float:
            action_mult_map = {
                "OPEN": self.strategy.slippage_multiplier_open,
                "ADDON": self.strategy.slippage_multiplier_addon,
                "REDUCE": self.strategy.slippage_multiplier_reduce,
                "EXIT": self.strategy.slippage_multiplier_exit,
            }
            regime_mult_map = {
                REGIME_TREND_CLEAN: self.strategy.slippage_multiplier_trend_clean,
                REGIME_TREND_NOISY: self.strategy.slippage_multiplier_trend_noisy,
                REGIME_MEAN_REVERT: self.strategy.slippage_multiplier_mean_revert,
                REGIME_SHOCK: self.strategy.slippage_multiplier_shock,
            }
            slippage_bricks = (
                self.strategy.base_slippage_bricks
                * action_mult_map.get(action, 1.0)
                * regime_mult_map.get(regime, 1.0)
            )
            return max(slippage_bricks, 0.05)

        def get_order_price_limits(
            self,
            intent: str,
            ref_price: float,
            action: str,
            regime: str
        ) -> Dict[str, float]:
            slippage_bricks = self.calculate_dynamic_slippage(action, regime)
            slippage_price = slippage_bricks * self.strategy.brick_size

            if intent in [INTENT_OPEN_LONG, INTENT_CLOSE_SHORT]:
                price_ceiling = ref_price + slippage_price
                limit_price = ref_price - self.strategy.price_tick * 2
                stop_limit_price = price_ceiling
            else:
                price_floor = ref_price - slippage_price
                limit_price = ref_price + self.strategy.price_tick * 2
                stop_limit_price = price_floor

            return {
                "limit_price": self.strategy._normalize_order_price(limit_price, intent),  # [FIX: C01]
                "price_limit": self.strategy._normalize_order_price(stop_limit_price, intent),  # [FIX: C01]
                "acceptable_slippage_bricks": slippage_bricks,
                "acceptable_slippage_price": slippage_price,
            }

        def should_use_market_order(self, intent: str, desired_price: float) -> bool:
            if not hasattr(self.strategy, "tick") or self.strategy.tick is None:
                return False

            tick = self.strategy.tick

            if intent in [INTENT_OPEN_LONG, INTENT_CLOSE_SHORT]:
                estimated_fill = tick.ask_price_1 if tick.ask_price_1 > 0 else desired_price
                slippage = estimated_fill - desired_price
            else:
                estimated_fill = tick.bid_price_1 if tick.bid_price_1 > 0 else desired_price
                slippage = desired_price - estimated_fill

            slippage_bricks = slippage / self.strategy.brick_size
            return slippage_bricks > self.strategy.max_slippage_for_limit

        def smart_order_submit(
            self,
            intent: str,
            volume: float,
            action: str,
            reason_code: str = "",
            extra: Optional[Dict[str, Any]] = None,
        ) -> List[str]:
            strategy = self.strategy
            regime = strategy.current_regime
            ref_price = strategy.last_price_seen or (strategy.last_brick_close or 0.0)
            price_limits = self.get_order_price_limits(intent, ref_price, action, regime)

            order_type = OrderType.LIMIT
            price = price_limits["limit_price"]

            if (
                self.should_use_market_order(intent, price_limits["limit_price"])
                or (regime == REGIME_SHOCK and action in ["EXIT", "REDUCE"])
            ):
                order_type = OrderType.MARKET
                price = 0

            if intent == INTENT_OPEN_LONG:
                order_ids = strategy.buy(price=price, volume=volume, order_type=order_type)
            elif intent == INTENT_OPEN_SHORT:
                order_ids = strategy.short(price=price, volume=volume, order_type=order_type)
            elif intent == INTENT_CLOSE_LONG:
                order_ids = strategy.sell(price=price, volume=volume, order_type=order_type)
            elif intent == INTENT_CLOSE_SHORT:
                order_ids = strategy.cover(price=price, volume=volume, order_type=order_type)
            else:
                strategy.write_log(f"未知下单意图: {intent}", level=logging.ERROR)
                return []

            meta = {
                "intent": intent,
                "action": action,
                "reason_code": reason_code,
                "price_limits": price_limits,
                "submit_time": time.time(),
                "timeout_ms": strategy.limit_order_timeout_ms,
                "cancel_requested": False,
                "fallback_after_cancel": False,
                "fallback_submitted": False,
                "extra": extra or {},
            }

            for oid in order_ids:
                self.active_orders[oid] = dict(meta)
                self.order_metas[oid] = dict(meta)

            strategy.write_cta_log(
                f"智能下单: action={action}, intent={intent}, "
                f"type={order_type.name}, price={price if price > 0 else 'MARKET'}, "
                f"volume={volume:.6f}, reason={reason_code}"
            )
            return order_ids

        def check_order_timeout(self, vt_orderid: str) -> bool:
            info = self.active_orders.get(vt_orderid)
            if not info:
                return False
            submit_time = info.get("submit_time")
            timeout_ms = info.get("timeout_ms", 500)
            if submit_time and (time.time() - submit_time) * 1000 > timeout_ms:
                return True
            return False

        def request_cancel_and_fallback(self, order: OrderData):
            info = self.active_orders.get(order.vt_orderid)
            if not info:
                return
            if info.get("cancel_requested"):
                return

            info["cancel_requested"] = True
            info["fallback_after_cancel"] = (
                self.strategy.enable_market_fallback
                and self.strategy.fallback_to_market_after_timeout
            )
            self.strategy.cancel_order(order.vt_orderid)

        def confirm_cancel_and_schedule_fallback(self, order: OrderData):
            info = self.order_metas.get(order.vt_orderid)
            if not info:
                return
            if not info.get("fallback_after_cancel"):
                return
            if info.get("fallback_submitted"):
                return

            remaining = float(order.volume - order.traded)
            if remaining > 0:
                self.pending_market_fallback.append(
                    (order.vt_orderid, info["intent"], remaining, info["action"])
                )
                info["fallback_submitted"] = True

        def remove_active(self, vt_orderid: str):
            self.active_orders.pop(vt_orderid, None)
            
        # [FIX: A04] 新增：扫描并标记超时订单，但不立即撤单
        def _scan_active_orders_for_timeout(self):
            """扫描并标记超时订单，但不立即撤单"""
            current_time = time.time()
            for vt_orderid, info in list(self.active_orders.items()):
                if vt_orderid in self._timeout_flagged_orders:
                    continue
                    
                submit_time = info.get("submit_time", 0)
                timeout_ms = info.get("timeout_ms", 500)
                
                if submit_time and (current_time - submit_time) * 1000 > timeout_ms:
                    # 标记为超时，但不立即撤单
                    info["timeout_marked"] = True
                    info["should_fallback_after_cancel"] = (
                        self.strategy.enable_market_fallback 
                        and self.strategy.fallback_to_market_after_timeout
                    )
                    self._timeout_flagged_orders.add(vt_orderid)
                    
                    self.strategy.write_log(
                        f"限价单超时标记: {vt_orderid}，等待on_order处理",
                        level=logging.INFO
                    )
        
        # [FIX: A04] 新增：在策略的on_order中调用此方法
        def on_order_callback(self, order: OrderData):
            """在策略的on_order中调用此方法"""
            vt_orderid = order.vt_orderid
            
            # 如果订单被标记为超时且仍在活跃状态，发起撤单
            if vt_orderid in self._timeout_flagged_orders and order.is_active():
                self.strategy.cancel_order(vt_orderid)
                self._timeout_flagged_orders.remove(vt_orderid)
                
                # 记录已请求撤单
                if vt_orderid in self.order_metas:
                    self.order_metas[vt_orderid]["cancel_requested"] = True
            
            # 如果订单已完成（非活跃），清理相关状态
            if not order.is_active():
                info = self.order_metas.get(vt_orderid)
                if info and info.get("timeout_marked") and info.get("should_fallback_after_cancel"):
                    # 处理回退逻辑
                    remaining = float(order.volume - order.traded)
                    if remaining > 0:
                        self.pending_market_fallback.append(
                            (vt_orderid, info["intent"], remaining, info["action"])
                        )
                
                # 清理状态
                self._timeout_flagged_orders.discard(vt_orderid)
                self.remove_active(vt_orderid)

    # ============================================================
    # 初始化
    # ============================================================
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 1秒驱动
        self.bg = BarGenerator(
            on_bar=self.on_bar,
            window=1,
            on_window_bar=self.on_one_sec_bar,
            interval=Interval.SECOND,
        )

        # 滑点与订单控制器
        self.slippage_ctrl = self.SlippageController(self)
        
        # [FIX: A03+A05] 新增：内部持仓跟踪器
        if self.enable_ledger_fix:
            self.ledger = PositionLedger(contract_size=self.contract_size)
        else:
            # 保持原有逻辑
            self.avg_entry_price = 0.0

        # Renko 状态
        self.bricks: List[RenkoBrick] = []
        self.renko_closes: Deque[float] = deque(
            maxlen=max(self.hurst_lookback_bricks + 5, self.regime_lookback_bricks + 5)
        )
        self.last_price_seen: Optional[float] = None
        self.last_brick_close: Optional[float] = None
        self.brick_id_counter = 0
        self.run_id_counter = 0
        self.current_run_dir = 0
        self.current_run_length = 0
        self.same_dir_count = 0
        self.recent_brick_timestamps: Deque[int] = deque(maxlen=2000)
        
        # [FIX: B03] 批量砖生成处理
        self.last_bulk_brick_action_id = -1

        # 市场状态
        self.current_regime = REGIME_MEAN_REVERT
        self.candidate_regime = REGIME_MEAN_REVERT
        self.regime_hold_bricks = 0
        self.last_hurst_value = 0.5
        
        # [FIX: B07] Shock识别优化
        self.shock_enter_count = 0
        self.shock_exit_count = 0

        # 策略状态
        self.position_role = ROLE_NONE
        self.addon_stage = 0
        self.addon_total_qty = 0.0
        self.entry_anchor_price = 0.0
        self.last_addon_anchor_price = 0.0
        self.reduce_anchor_price = 0.0
        self.trend_has_advanced_since_addon = False
        
        # [FIX: A01] 减仓冷却
        self.reduce_cooldown_remaining = 0

        # 风控与批次
        self.hard_reversal_cooldown_remaining = 0
        self.consecutive_loss_batches = 0
        self.loss_pause_active = False
        self.loss_pause_remaining_bricks = 0
        self.current_batch: Optional[TradeBatch] = None
        self.last_action_brick_id = -1
        
        # [FIX: A06] 启动期保护
        self.warmup_complete = False

        # 估算价格与盈亏
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_bricks = 0.0
        self.last_tick_ts = 0

        # 其他
        self.total_new_bricks = 0
        self.tick: Optional[TickData] = None
        
        # 输出修复状态
        self._log_fix_status()

    def _log_fix_status(self):
        """输出当前启用的修复状态"""
        self.write_log("=" * 60)
        self.write_log("Renko趋势策略修复整合版 - 修复状态")
        self.write_log("=" * 60)
        
        fixes = [
            ("A01+A02", "减仓逻辑优化", self.enable_reduce_logic_fix),
            ("A03+A05", "内部持仓跟踪器", self.enable_ledger_fix),
            ("A04", "订单超时状态机", self.enable_order_timeout_fix),
            ("A06", "启动期保护", self.enable_warmup_protection),
            ("B01", "Regime识别滞后优化", self.enable_regime_lag_fix),
            ("B03", "批量砖生成处理", self.enable_bulk_brick_fix),
            ("B05", "Hard reversal冷却完善", self.enable_hard_reversal_fix),
            ("B07", "Shock识别优化", self.enable_shock_detection_fix),
            ("C01", "价格归一化方向处理", self.enable_price_normalization_fix),
            ("C02", "参数校验增强", self.enable_param_validation_fix),
            ("C07", "重复扫描修复", self.enable_duplicate_scan_fix),
            ("EXT", "历史数据加载", self.enable_history_loading),
            ("EXT", "资金管理模型增强", self.enable_risk_based_sizing),
        ]
        
        for code, desc, enabled in fixes:
            status = "✅ 启用" if enabled else "❌ 禁用"
            self.write_log(f"{code:8} {desc:20} {status}")
        
        self.write_log("=" * 60)

    def on_init(self):
        self.write_log("策略初始化")
        self._validate_params()
        
        # [FIX: 历史数据加载] 在初始化时加载历史数据
        if self.enable_history_loading and self.engine:
            self._load_history_bars()
        
        # [FIX: A06] 检查预热状态
        if self.enable_warmup_protection:
            if len(self.bricks) >= self.min_bricks_for_trading:
                self.warmup_complete = True
                self.write_log(f"历史数据充足，预热完成，已收集{len(self.bricks)}个砖块")
            else:
                self.write_log(f"预热中，已收集{len(self.bricks)}个砖块，需要{self.min_bricks_for_trading}个")

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    # ============================================================
    # 行情驱动
    # ============================================================
    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)
        self.last_price_seen = tick.last_price
        self.tick = tick

    def on_bar(self, bar: BarData):
        self.bg.update_bar(bar)

    def on_one_sec_bar(self, bar: BarData):
        # [FIX: C07] 只保留一个主扫描入口
        internal_bar = {
            "ts": int(bar.datetime.timestamp() * 1000),
            "open": bar.open_price,
            "high": bar.high_price,
            "low": bar.low_price,
            "close": bar.close_price,
            "volume": bar.volume,
        }
        self._process_bar(internal_bar)
        
        # 扫描超时订单
        if self.enable_order_timeout_fix:
            self.slippage_ctrl._scan_active_orders_for_timeout()
        else:
            # 原有逻辑
            self._scan_active_orders_for_timeout()
            
        self._process_pending_fallbacks()
        self.put_event()
        
    # [FIX: 历史数据加载] 新增：加载历史K线
    def _load_history_bars(self):
        """加载历史K线数据初始化Renko图表"""
        try:
            # 计算需要加载的历史数据长度
            # 我们需要足够的砖块来覆盖所有分析窗口
            max_lookback = max(
                self.regime_lookback_bricks,
                self.hurst_lookback_bricks,
                self.min_bricks_for_trading
            )
            
            # 估计需要的K线数量：每个砖块至少需要1根K线，乘以安全系数
            estimated_bars_needed = max_lookback * 3
            
            # 计算开始时间
            end_date = datetime.now()
            # 假设平均每秒生成0.1个砖块，保守估计
            seconds_needed = estimated_bars_needed * 10
            start_date = end_date - timedelta(seconds=seconds_needed)
            
            self.write_log(f"开始加载历史数据: {start_date} 到 {end_date}")
            
            # 加载历史K线
            bars = self.load_bar(
                self.vt_symbol,
                interval=Interval.SECOND,
                start=start_date,
                end=end_date,
                use_database=True
            )
            
            if bars:
                self.write_log(f"成功加载 {len(bars)} 根历史K线")
                
                # 处理历史K线生成砖块
                history_brick_count = 0
                for bar in bars:
                    internal_bar = {
                        "ts": int(bar.datetime.timestamp() * 1000),
                        "open": bar.open_price,
                        "high": bar.high_price,
                        "low": bar.low_price,
                        "close": bar.close_price,
                        "volume": bar.volume,
                    }
                    
                    # 使用专门的历史数据处理函数
                    new_bricks = self._process_bar_to_bricks_for_init(internal_bar)
                    for brick in new_bricks:
                        self.bricks.append(brick)
                        self.renko_closes.append(brick.close_price)
                        self._update_run_state_after_brick_for_init(brick)
                        history_brick_count += 1
                
                self.write_log(f"历史数据生成 {history_brick_count} 个Renko砖块")
                self.write_log(f"当前砖块总数: {len(self.bricks)}")
                
                # 如果有足够的历史砖块，预热完成
                if len(self.bricks) >= self.min_bricks_for_trading:
                    self.warmup_complete = True
                    
            else:
                self.write_log("未加载到历史数据，将从实时数据开始构建")
                
        except Exception as e:
            self.write_log(f"历史数据加载失败: {e}", level=logging.ERROR)
            
    # [FIX: 历史数据加载] 新增：历史数据处理函数
    def _process_bar_to_bricks_for_init(self, bar: Dict[str, Any]) -> List[RenkoBrick]:
        """处理历史K线生成砖块（不触发交易逻辑）"""
        new_bricks: List[RenkoBrick] = []
        close_price = bar["close"]
        
        if self.last_brick_close is None:
            self.last_brick_close = close_price
            self.renko_closes.append(close_price)
            return new_bricks
        
        price_diff = close_price - self.last_brick_close
        n = int(abs(price_diff) // self.brick_size)
        if n <= 0:
            return new_bricks
        
        n = min(n, self.max_bricks_per_bar)
        direction = 1 if price_diff > 0 else -1
        generated_count = n
        
        for _ in range(n):
            open_price = self.last_brick_close
            close_p = open_price + direction * self.brick_size
            
            if direction != self.current_run_dir:
                self.run_id_counter += 1
                self.current_run_dir = direction
                self.current_run_length = 1
            else:
                self.current_run_length += 1
            
            self.brick_id_counter += 1
            brick = RenkoBrick(
                brick_id=self.brick_id_counter,
                ts=bar["ts"],
                open_price=open_price,
                close_price=close_p,
                direction=direction,
                source_price=bar["close"],
                source_tick_ts=bar["ts"],
                bricks_generated_in_tick=generated_count,
                run_id=self.run_id_counter,
            )
            
            self.last_brick_close = close_p
            new_bricks.append(brick)
        
        return new_bricks
        
    # [FIX: 历史数据加载] 新增：更新运行状态（历史数据）
    def _update_run_state_after_brick_for_init(self, brick: RenkoBrick):
        """更新运行状态（用于历史数据处理）"""
        if brick.direction == self.current_run_dir:
            self.current_run_length += 1
        else:
            self.current_run_dir = brick.direction
            self.current_run_length = 1

    def _process_bar(self, bar: Dict[str, Any]):
        self.last_tick_ts = bar["ts"]
        self.last_price_seen = bar["close"]

        if self.log_tick_events and self._should_sample_tick_log():
            self.write_cta_log(
                f"1秒Bar: 开={bar['open']}, 高={bar['high']}, 低={bar['low']}, 收={bar['close']}"
            )

        new_bricks = self._process_bar_to_bricks(bar)

        if not new_bricks:
            self._update_unrealized_pnl_with_last_price()
            return

        for brick in new_bricks:
            self.total_new_bricks += 1
            self._on_new_brick(brick)

    def _process_bar_to_bricks(self, bar: Dict[str, Any]) -> List[RenkoBrick]:
        new_bricks: List[RenkoBrick] = []
        close_price = bar["close"]

        if self.last_brick_close is None:
            self.last_brick_close = close_price
            self.renko_closes.append(close_price)
            return new_bricks

        price_diff = close_price - self.last_brick_close
        n = int(abs(price_diff) // self.brick_size)
        if n <= 0:
            return new_bricks

        n = min(n, self.max_bricks_per_bar)
        direction = 1 if price_diff > 0 else -1
        generated_count = n

        if generated_count >= self.dislocated_single_bar_bricks:
            self.write_log(
                f"砖块批量生成警告: 数量={generated_count}, 方向={direction}, "
                f"起始价={self.last_brick_close}",
                level=logging.WARNING,
            )

        for _ in range(n):
            open_price = self.last_brick_close
            close_p = open_price + direction * self.brick_size

            if direction != self.current_run_dir:
                self.run_id_counter += 1

            self.brick_id_counter += 1
            brick = RenkoBrick(
                brick_id=self.brick_id_counter,
                ts=bar["ts"],
                open_price=open_price,
                close_price=close_p,
                direction=direction,
                source_price=bar["close"],
                source_tick_ts=bar["ts"],
                bricks_generated_in_tick=generated_count,
                run_id=self.run_id_counter,
            )

            self.last_brick_close = close_p
            self.bricks.append(brick)
            self.renko_closes.append(close_p)
            self.recent_brick_timestamps.append(bar["ts"])
            new_bricks.append(brick)

        return new_bricks

    def _scan_active_orders_for_timeout(self):
        # [FIX: A04] 如果启用了修复，使用新的扫描逻辑
        if self.enable_order_timeout_fix:
            self.slippage_ctrl._scan_active_orders_for_timeout()
            return
            
        # 原有逻辑
        for vt_orderid, info in list(self.slippage_ctrl.active_orders.items()):
            if info.get("cancel_requested"):
                continue
            if self.slippage_ctrl.check_order_timeout(vt_orderid):
                self.write_log(
                    f"限价单超时，准备撤单: {vt_orderid}",
                    level=logging.INFO,
                )
                # [FIX: 删除残留的mock_order代码]
                self.cancel_order(vt_orderid)
                info["cancel_requested"] = True
                info["fallback_after_cancel"] = (
                    self.enable_market_fallback and self.fallback_to_market_after_timeout
                )

    def _process_pending_fallbacks(self):
        if not self.slippage_ctrl.pending_market_fallback:
            return

        tasks = list(self.slippage_ctrl.pending_market_fallback)
        self.slippage_ctrl.pending_market_fallback.clear()

        for vt_orderid, intent, volume, action in tasks:
            self.write_log(
                f"执行市价回退: 原单={vt_orderid}, intent={intent}, volume={volume}, action={action}"
            )

            if intent == INTENT_OPEN_LONG:
                self.buy(price=0, volume=volume, order_type=OrderType.MARKET)
            elif intent == INTENT_OPEN_SHORT:
                self.short(price=0, volume=volume, order_type=OrderType.MARKET)
            elif intent == INTENT_CLOSE_LONG:
                self.sell(price=0, volume=volume, order_type=OrderType.MARKET)
            elif intent == INTENT_CLOSE_SHORT:
                self.cover(price=0, volume=volume, order_type=OrderType.MARKET)

    # ============================================================
    # 订单与成交回调
    # ============================================================
    def on_order(self, order: OrderData):
        super().on_order(order)
        
        # [FIX: A04] 如果启用了修复，使用新的回调处理
        if self.enable_order_timeout_fix:
            self.slippage_ctrl.on_order_callback(order)
        else:
            # 原有逻辑
            if not order.is_active():
                info = self.slippage_ctrl.order_metas.get(order.vt_orderid)
                if info and info.get("cancel_requested"):
                    if order.status in [
                        Status.CANCELLED,
                        Status.PARTTRADED,
                        Status.REJECTED,
                    ] or order.traded < order.volume:
                        self.slippage_ctrl.confirm_cancel_and_schedule_fallback(order)

                self.slippage_ctrl.remove_active(order.vt_orderid)
                return

            # 若订单仍活跃，也可以在这里进行一次保险超时处理
            if order.type == OrderType.LIMIT:
                if self.slippage_ctrl.check_order_timeout(order.vt_orderid):
                    self.write_log(
                        f"限价单超时: {order.vt_orderid}, 已成交={order.traded}/{order.volume}",
                        level=logging.INFO,
                    )
                    self.slippage_ctrl.request_cancel_and_fallback(order)

        if self.log_order_events:
            self.write_cta_log(
                f"订单更新: {order.vt_orderid}, 状态={order.status}, "
                f"已成交={order.traded}/{order.volume}, 类型={order.type}"
            )

    def on_trade(self, trade: TradeData):
        if self.log_order_events:
            self.write_cta_log(
                f"成交: {trade.vt_tradeid}, order={trade.vt_orderid}, "
                f"方向={trade.direction}, 数量={trade.volume}, 价格={trade.price}, 开平={trade.offset}"
            )

        self._apply_trade_to_strategy_state(trade)
        self._update_unrealized_pnl_with_last_price()
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass

    # ============================================================
    # 核心策略逻辑实现 - 砖块处理与状态更新
    # ============================================================
    def _on_new_brick(self, brick: RenkoBrick):
        # [FIX: A06] 启动期保护
        if self.enable_warmup_protection and not self.warmup_complete:
            if len(self.bricks) >= self.min_bricks_for_trading:
                self.warmup_complete = True
                self.write_log(f"预热完成，已收集{len(self.bricks)}个砖块，开始正常交易")
            else:
                # 预热期间，只更新状态，不执行交易
                self._update_run_state_after_brick(brick)
                self._update_unrealized_pnl_with_last_price()
                if self.log_brick_events:
                    self.write_cta_log(
                        f"预热中: 砖块{brick.brick_id}, 剩余{self.min_bricks_for_trading - len(self.bricks)}个"
                    )
                return
        
        # [FIX: B03] 批量砖生成处理
        if self.enable_bulk_brick_fix and brick.bricks_generated_in_tick > 1:
            if brick.brick_id <= self.last_bulk_brick_action_id + self.bulk_brick_action_cooldown:
                # 跳过冷却期内的动作
                self._update_run_state_after_brick(brick)
                self._update_unrealized_pnl_with_last_price()
                return
            self.last_bulk_brick_action_id = brick.brick_id

        self._update_run_state_after_brick(brick)

        if self.log_brick_events:
            self.write_cta_log(
                f"砖块生成: ID={brick.brick_id}, 开={brick.open_price:.1f}, "
                f"收={brick.close_price:.1f}, 方向={brick.direction}, 同向计数={self.same_dir_count}"
            )

        self._update_unrealized_pnl_with_last_price()

        if self.current_batch:
            self.current_batch.holding_bricks += 1
            self.current_batch.regime_history.append(self.current_regime)
            self._update_trade_batch_excursions()

        regime_features = self._compute_regime_features(brick)

        prev_regime = self.current_regime
        final_regime, decision_meta = self._classify_market_regime(regime_features)
        self.current_regime = final_regime

        if final_regime == prev_regime:
            self.regime_hold_bricks += 1
        else:
            self.regime_hold_bricks = 1
            self.write_log(
                f"市场状态切换: {prev_regime} -> {final_regime}, "
                f"原因: {decision_meta.get('reason_codes', [])}"
            )

        if self.hard_reversal_cooldown_remaining > 0:
            self.hard_reversal_cooldown_remaining -= 1

        if self.loss_pause_active and self.loss_pause_remaining_bricks > 0:
            self.loss_pause_remaining_bricks -= 1
            if self.loss_pause_remaining_bricks <= 0:
                self.loss_pause_active = False
                self.write_log("连续亏损暂停期结束，恢复允许新开仓")
                
        # [FIX: A01] 减仓冷却更新
        if self.reduce_cooldown_remaining > 0:
            self.reduce_cooldown_remaining -= 1

        ctx = self._build_decision_context(regime_features)
        self._evaluate_and_execute(brick, regime_features, ctx)

    def _update_run_state_after_brick(self, brick: RenkoBrick):
        if brick.direction == self.current_run_dir:
            self.current_run_length += 1
            self.same_dir_count += 1
        else:
            if self.current_run_dir != 0 and self.log_structure_snapshots:
                self.write_cta_log(
                    f"RUN结束: 方向={self.current_run_dir}, 长度={self.current_run_length}"
                )
            self.current_run_dir = brick.direction
            self.current_run_length = 1
            self.same_dir_count = 1
            if self.log_structure_snapshots:
                self.write_cta_log(
                    f"RUN开始: 方向={self.current_run_dir}, 起始价={brick.open_price}"
                )

        if self.same_dir_count in [2, 3] and self.log_structure_snapshots:
            self.write_cta_log(f"连续同向砖: {self.same_dir_count}根, 方向={brick.direction}")

    # ============================================================
    # 市场状态识别
    # ============================================================
    def _compute_structure_features(self) -> StructureFeatures:
        recent = self.bricks[-self.regime_lookback_bricks:]
        feat = StructureFeatures()

        if len(recent) < 2:
            feat.current_run_length = self.current_run_length
            return feat

        dirs = [b.direction for b in recent]
        n = len(dirs)

        same_transitions = 0
        reversals = 0
        for i in range(1, n):
            if dirs[i] == dirs[i - 1]:
                same_transitions += 1
            else:
                reversals += 1

        feat.directional_persistence = same_transitions / max(n - 1, 1)
        feat.reversal_density = reversals / max(n - 1, 1)

        run_lengths = []
        cur_len = 1
        for i in range(1, n):
            if dirs[i] == dirs[i - 1]:
                cur_len += 1
            else:
                run_lengths.append(cur_len)
                cur_len = 1
        run_lengths.append(cur_len)

        feat.avg_run_length = self._safe_mean(run_lengths, default=1.0)
        feat.max_run_length = max(run_lengths) if run_lengths else 1

        up_count = sum(1 for d in dirs if d > 0)
        down_count = sum(1 for d in dirs if d < 0)
        feat.up_count_window = up_count
        feat.down_count_window = down_count
        feat.net_displacement_efficiency = abs(up_count - down_count) / max(n, 1)

        opposite = down_count if up_count >= down_count else up_count
        feat.pullback_share = opposite / max(n, 1)

        feat.current_run_length = self.current_run_length
        feat.extension_score = self.current_run_length / max(feat.avg_run_length, 1e-9)
        feat.chop_score = feat.reversal_density * (1 - feat.net_displacement_efficiency)

        if (
            self.log_structure_snapshots
            and self.total_new_bricks % self.snapshot_log_every_n_bricks == 0
        ):
            self.write_cta_log(
                f"结构快照: 方向持续={feat.directional_persistence:.3f}, "
                f"反转密度={feat.reversal_density:.3f}, 平均run长度={feat.avg_run_length:.2f}, "
                f"净推进效率={feat.net_displacement_efficiency:.3f}"
            )

        return feat

    def _calc_rolling_hurst_on_renko_close(self) -> float:
        series = list(self.renko_closes)[-self.hurst_lookback_bricks:]
        if len(series) < 16:
            return 0.5

        x = [series[i] - series[i - 1] for i in range(1, len(series))]
        if len(x) < 8:
            return 0.5

        mean_x = self._safe_mean(x)
        dev = [v - mean_x for v in x]

        cum = []
        s = 0.0
        for v in dev:
            s += v
            cum.append(s)

        r = (max(cum) - min(cum)) if cum else 0.0
        sd = statistics.pstdev(x) if len(x) >= 2 else 0.0

        if sd <= 1e-12 or r <= 1e-12:
            return 0.5

        rs = r / sd
        n = len(x)
        h = math.log(max(rs, 1e-12)) / math.log(max(n, 2))
        return self._clamp(h, 0.0, 1.0)

    def _calc_tempo_state(self) -> Tuple[str, int]:
        cur_ts = self.last_tick_ts
        while (
            self.recent_brick_timestamps
            and cur_ts - self.recent_brick_timestamps[0] > self.tempo_window_seconds * 1000
        ):
            self.recent_brick_timestamps.popleft()

        cnt = len(self.recent_brick_timestamps)
        if cnt <= self.tempo_slow_max:
            return "SLOW", cnt
        if cnt <= self.tempo_normal_max:
            return "NORMAL", cnt
        if cnt <= self.tempo_fast_max:
            return "FAST", cnt
        return "BURST", cnt

    def _detect_structure_break(self, structure: StructureFeatures) -> bool:
        return (
            structure.reversal_density >= 0.50
            and structure.chop_score >= 0.30
            and structure.extension_score >= 1.8
        )

    def _compute_regime_features(self, latest_brick: RenkoBrick) -> Dict[str, Any]:
        structure = self._compute_structure_features()
        hurst_value = self._calc_rolling_hurst_on_renko_close() if self.enable_hurst else 0.5
        tempo_state, tempo_brick_count = self._calc_tempo_state()
        dislocated_tag = latest_brick.bricks_generated_in_tick >= self.dislocated_single_bar_bricks
        structure_break = self._detect_structure_break(structure)
        hurst_deteriorated = (self.last_hurst_value - hurst_value) >= 0.08
        self.last_hurst_value = hurst_value

        shock_score_raw = 0
        if tempo_state == "BURST":
            shock_score_raw += 1
        if dislocated_tag:
            shock_score_raw += 2
        if structure.extension_score >= self.shock_extension_threshold:
            shock_score_raw += 1
        if structure_break:
            shock_score_raw += 1
        if hurst_deteriorated:
            shock_score_raw += 1

        if (
            self.log_regime_snapshots
            and self.total_new_bricks % self.snapshot_log_every_n_bricks == 0
        ):
            self.write_cta_log(
                f"状态特征: Hurst={hurst_value:.3f}, Tempo={tempo_state}, "
                f"错位标签={dislocated_tag}, 冲击分={shock_score_raw}"
            )

        return {
            "structure": structure,
            "hurst_value": hurst_value,
            "tempo_state": tempo_state,
            "tempo_brick_count": tempo_brick_count,
            "dislocated_tag": dislocated_tag,
            "shock_score_raw": shock_score_raw,
            "structure_break": structure_break,
            "hurst_deteriorated": hurst_deteriorated,
        }

    def _classify_market_regime(self, rf: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        structure: StructureFeatures = rf["structure"]
        hurst_value = rf["hurst_value"]
        tempo_state = rf["tempo_state"]
        dislocated_tag = rf["dislocated_tag"]
        shock_score_raw = rf["shock_score_raw"]
        prev_regime = self.current_regime
        candidate = REGIME_MEAN_REVERT
        reason_codes: List[str] = []
        
        # [FIX: B07] Shock识别优化
        if self.enable_shock_detection_fix:
            # 检查是否需要进入SHOCK状态
            shock_condition = (
                shock_score_raw >= self.shock_score_threshold
                or (dislocated_tag and tempo_state in ["FAST", "BURST"])
                or (tempo_state == "BURST" and structure.extension_score >= self.shock_extension_threshold)
            )
            
            if shock_condition:
                self.shock_enter_count += 1
                self.shock_exit_count = 0
            else:
                self.shock_exit_count += 1
                self.shock_enter_count = 0
                
            # 需要连续确认才切换
            if prev_regime != REGIME_SHOCK and self.shock_enter_count >= self.shock_enter_confirm_bricks:
                candidate = REGIME_SHOCK
                reason_codes.append(f"shock_override(confirmed:{self.shock_enter_count})")
            elif prev_regime == REGIME_SHOCK and self.shock_exit_count < self.shock_exit_confirm_bricks:
                candidate = REGIME_SHOCK
                reason_codes.append(f"shock_hold(exit_count:{self.shock_exit_count})")
            else:
                # 正常分类逻辑
                self._classify_normal_regime(structure, hurst_value, candidate, reason_codes)
        else:
            # 原有逻辑
            if (
                shock_score_raw >= self.shock_score_threshold
                or (dislocated_tag and tempo_state in ["FAST", "BURST"])
                or (tempo_state == "BURST" and structure.extension_score >= self.shock_extension_threshold)
            ):
                candidate = REGIME_SHOCK
                reason_codes.append("shock_override")
            else:
                self._classify_normal_regime(structure, hurst_value, candidate, reason_codes)

        final_regime = candidate

        if candidate == REGIME_SHOCK and self.shock_override_hold:
            final_regime = REGIME_SHOCK
        else:
            # [FIX: B01] Regime切换双阈值机制
            if self.enable_regime_lag_fix:
                if prev_regime != candidate:
                    if prev_regime == REGIME_TREND_CLEAN and candidate == REGIME_TREND_CLEAN:
                        # 进入清晰趋势
                        required_hold = self.trend_clean_enter_hold_bricks
                    elif prev_regime == REGIME_TREND_CLEAN and candidate != REGIME_TREND_CLEAN:
                        # 退出清晰趋势
                        required_hold = self.trend_clean_exit_hold_bricks
                    else:
                        required_hold = self.min_regime_hold_bricks
                    
                    if self.regime_hold_bricks < required_hold:
                        final_regime = prev_regime
                        reason_codes.append(f"min_hold_blocked(need:{required_hold})")
            else:
                # 原有逻辑
                if prev_regime != candidate and self.regime_hold_bricks < self.min_regime_hold_bricks:
                    final_regime = prev_regime
                    reason_codes.append("min_hold_blocked")

        # 原有hysteresis逻辑
        if not self.enable_regime_lag_fix:
            if prev_regime == REGIME_TREND_CLEAN and candidate != REGIME_TREND_CLEAN:
                structure_trend_score = (
                    structure.directional_persistence * 0.30
                    + (1.0 - structure.reversal_density) * 0.20
                    + structure.net_displacement_efficiency * 0.25
                    + self._clamp(structure.avg_run_length / 3.0, 0.0, 1.0) * 0.10
                    + (1.0 - structure.pullback_share) * 0.10
                )
                if structure_trend_score >= self.trend_clean_exit_threshold:
                    final_regime = REGIME_TREND_CLEAN
                    reason_codes.append("hysteresis_hold_clean")

        self.candidate_regime = candidate

        if self.log_regime_snapshots:
            self.write_cta_log(
                f"状态决策: 候选={candidate}, 最终={final_regime}, 原因={reason_codes}"
            )

        return final_regime, {
            "reason_codes": reason_codes,
            "hold_bricks_before_change": self.regime_hold_bricks,
        }
        
    def _classify_normal_regime(self, structure: StructureFeatures, hurst_value: float, 
                               candidate: str, reason_codes: List[str]):
        """正常市场状态分类逻辑"""
        trend_score = 0.0
        chop_score = 0.0

        trend_score += structure.directional_persistence * 0.30
        trend_score += (1.0 - structure.reversal_density) * 0.20
        trend_score += structure.net_displacement_efficiency * 0.25
        trend_score += self._clamp(structure.avg_run_length / 3.0, 0.0, 1.0) * 0.10
        trend_score += (1.0 - structure.pullback_share) * 0.10
        trend_score += self._clamp(
            1.0 - max(structure.extension_score - 2.5, 0.0) / 3.0, 0.0, 1.0
        ) * 0.05

        chop_score += structure.reversal_density * 0.35
        chop_score += (1.0 - structure.net_displacement_efficiency) * 0.25
        chop_score += self._clamp(1.0 - structure.avg_run_length / 3.0, 0.0, 1.0) * 0.15
        chop_score += structure.pullback_share * 0.15
        chop_score += structure.chop_score * 0.10

        if hurst_value >= self.hurst_trend_threshold:
            trend_score += 0.08
        elif hurst_value <= self.hurst_mean_revert_threshold:
            chop_score += 0.08

        if (
            structure.directional_persistence >= self.persistence_trend_clean_min
            and structure.reversal_density <= self.reversal_trend_clean_max
            and structure.net_displacement_efficiency >= self.efficiency_trend_clean_min
            and structure.pullback_share <= self.pullback_trend_clean_max
            and hurst_value >= self.hurst_trend_threshold
            and trend_score >= self.trend_clean_enter_threshold
        ):
            candidate = REGIME_TREND_CLEAN
            reason_codes.extend(["structure_trend_strong", "hurst_confirm"])
        elif (
            structure.directional_persistence >= self.persistence_trend_noisy_min
            and structure.net_displacement_efficiency >= self.efficiency_trend_noisy_min
            and structure.avg_run_length >= self.avg_run_trend_noisy_min
            and hurst_value > self.hurst_mean_revert_threshold
            and trend_score > chop_score
        ):
            candidate = REGIME_TREND_NOISY
            reason_codes.extend(["structure_trend_ok", "hurst_not_weak"])
        else:
            candidate = REGIME_MEAN_REVERT
            reason_codes.append("chop_dominant_or_trend_weak")

    def _build_decision_context(self, rf: Dict[str, Any]) -> DecisionContext:
        # [FIX: A03+A05] 使用内部持仓跟踪器
        if self.enable_ledger_fix:
            net_qty = self.ledger.net_qty
            avg_price = self.ledger.avg_entry_price
        else:
            net_qty = self.pos
            avg_price = self.avg_entry_price
            
        if net_qty > 0:
            pos_state_str = POSITION_LONG
        elif net_qty < 0:
            pos_state_str = POSITION_SHORT
        else:
            pos_state_str = POSITION_FLAT

        ctx = DecisionContext(
            ts=int(time.time() * 1000),
            position_state=pos_state_str,
            position_role=self.position_role,
            net_position_qty=abs(net_qty),
            avg_entry_price=avg_price,
            unrealized_pnl=self.unrealized_pnl,
            unrealized_pnl_bricks=self.unrealized_pnl_bricks,
            addon_stage=self.addon_stage,
            addon_total_qty=self.addon_total_qty,
            entry_anchor_price=self.entry_anchor_price,
            last_addon_anchor_price=self.last_addon_anchor_price,
            reduce_anchor_price=self.reduce_anchor_price,
            regime=self.current_regime,
            tempo_state=rf["tempo_state"],
            loss_pause_active=self.loss_pause_active,
            hard_reversal_ready=self._check_hard_reversal_condition(rf),
        )

        if self.log_decision_actions:
            self.write_cta_log(
                f"决策上下文: 持仓状态={ctx.position_state}, 角色={ctx.position_role}, "
                f"市场状态={ctx.regime}, 加仓阶段={ctx.addon_stage}"
            )

        return ctx

    # ============================================================
    # 交易信号检查
    # ============================================================
    def _evaluate_and_execute(self, brick: RenkoBrick, rf: Dict[str, Any], ctx: DecisionContext):
        if self.last_action_brick_id == brick.brick_id:
            return

        # [FIX: B05] Hard reversal冷却行为完善
        if self.enable_hard_reversal_fix and self.hard_reversal_full_cooldown:
            if self.hard_reversal_cooldown_remaining > 0:
                # 冷却期内禁止任何开仓
                if self.pos == 0:
                    if self.log_decision_actions:
                        self.write_cta_log(f"硬反转冷却中，剩余{self.hard_reversal_cooldown_remaining}砖块，禁止新开仓")
                    return
        else:
            # 原有逻辑：冷却期只阻止空仓后再开仓
            if self.pos == 0 and self.hard_reversal_cooldown_remaining > 0:
                return

        # 1) 先处理平仓
        if self.pos != 0:
            exit_reason = self._check_exit_signal(rf)
            if exit_reason:
                self._execute_full_exit(reason_code=exit_reason)
                self.last_action_brick_id = brick.brick_id
                return

        # 2) probe -> core
        if self.pos != 0 and self.position_role == ROLE_PROBE:
            if self._check_probe_to_core_upgrade():
                self._upgrade_probe_to_core()

        # 3) reduce addon
        if self.pos != 0 and self.addon_total_qty > 0:
            if self._check_reduce_signal_after_pullback():
                self._execute_reduce_addon(reason_code="PULLBACK_REDUCE_50")
                self.last_action_brick_id = brick.brick_id
                return

        # 4) addon
        if self.pos != 0:
            addon_stage_target = self._check_addon_signal(rf)
            if addon_stage_target > 0:
                self._execute_addon(addon_stage_target, rf)
                self.last_action_brick_id = brick.brick_id
                return

        # 5) 开仓
        if self.pos == 0:
            if self._check_loss_pause_gate():
                return
            if self.hard_reversal_cooldown_remaining > 0 and not self.enable_hard_reversal_fix:
                return
            open_side = self._check_open_signal()
            if open_side:
                self._execute_open_probe(open_side, rf)
                self.last_action_brick_id = brick.brick_id

    def _check_open_signal(self) -> Optional[str]:
        if len(self.bricks) < self.open_trigger_bricks:
            return None

        # [FIX: A06] 预热期保护
        if self.enable_warmup_protection and not self.warmup_complete:
            return None

        if self.enable_regime_filter:
            # [FIX: B01] 允许candidate_regime参与开仓评分
            if self.enable_regime_lag_fix:
                regime_for_check = self.candidate_regime
            else:
                regime_for_check = self.current_regime
                
            if regime_for_check not in [REGIME_TREND_CLEAN, REGIME_TREND_NOISY]:
                return None

        recent = self.bricks[-self.open_trigger_bricks:]
        dirs = [b.direction for b in recent]

        if all(d > 0 for d in dirs):
            if self.log_decision_actions:
                self.write_cta_log("开仓信号: LONG")
            return POSITION_LONG

        if all(d < 0 for d in dirs):
            if self.log_decision_actions:
                self.write_cta_log("开仓信号: SHORT")
            return POSITION_SHORT

        return None

    def _check_probe_to_core_upgrade(self) -> bool:
        if self.position_role != ROLE_PROBE:
            return False
        if self.unrealized_pnl_bricks >= self.core_upgrade_profit_bricks:
            if self.log_decision_actions:
                self.write_cta_log(
                    f"Probe升级Core: 浮盈砖数={self.unrealized_pnl_bricks:.2f}"
                )
            return True
        return False

    def _check_addon_signal(self, rf: Dict[str, Any]) -> int:
        if not self.enable_addon:
            return 0
        if self.pos == 0:
            return 0
        if self.position_role not in [ROLE_CORE, ROLE_PROBE]:
            return 0

        max_addons = self._get_max_addons_by_regime(self.current_regime)
        if self.addon_stage >= max_addons:
            return 0

        if not self._regime_allows_addon(self.current_regime, rf["hurst_value"]):
            return 0

        stage_to_enter = self.addon_stage + 1
        if stage_to_enter > 3:
            return 0

        required_same = self._get_required_same_dir_bricks_for_addon_stage(stage_to_enter)
        if self.same_dir_count < required_same:
            return 0

        anchor = self._get_addon_reference_anchor()
        displacement = self._calc_brick_displacement_from_anchor(anchor)
        if displacement < self.addon_displacement_bricks:
            return 0

        last_dir = self.bricks[-1].direction if self.bricks else 0
        if self.pos > 0 and last_dir <= 0:
            return 0
        if self.pos < 0 and last_dir >= 0:
            return 0

        if self.log_decision_actions:
            self.write_cta_log(
                f"加仓信号: stage={stage_to_enter}, same_dir={self.same_dir_count}, disp={displacement:.2f}"
            )
        return stage_to_enter

    def _check_reduce_signal_after_pullback(self) -> bool:
        """[FIX: A01] 修复reduce_on_pullback_bricks参数使用"""
        if self.addon_total_qty <= 0:
            return False
        if not self.trend_has_advanced_since_addon:
            return False
        if self.reduce_cooldown_remaining > 0:
            return False
        if len(self.bricks) < self.reduce_on_pullback_bricks:
            return False
        
        # 检查最近N根砖块是否连续反向
        recent_bricks = self.bricks[-self.reduce_on_pullback_bricks:]
        
        if self.pos > 0:  # 多头持仓
            if all(b.direction < 0 for b in recent_bricks):
                return True
        elif self.pos < 0:  # 空头持仓
            if all(b.direction > 0 for b in recent_bricks):
                return True
        
        return False

    def _check_hard_reversal_condition(self, rf: Dict[str, Any]) -> bool:
        if not self.enable_hard_reversal:
            return False

        extreme = (
            self.current_regime == REGIME_SHOCK
            or rf["dislocated_tag"]
            or rf["shock_score_raw"] >= self.shock_score_threshold
        )
        return extreme if self.hard_reversal_requires_extreme else True

    def _check_exit_signal(self, rf: Dict[str, Any]) -> Optional[str]:
        if self.pos == 0 or len(self.bricks) < self.close_trigger_bricks:
            return None

        recent = self.bricks[-self.close_trigger_bricks:]
        dirs = [b.direction for b in recent]

        if self.pos > 0 and all(d < 0 for d in dirs):
            if self._check_hard_reversal_condition(rf):
                return "HARD_REVERSAL_EXIT_LONG"
            return "EXIT_LONG_2BRICKS"

        if self.pos < 0 and all(d > 0 for d in dirs):
            if self._check_hard_reversal_condition(rf):
                return "HARD_REVERSAL_EXIT_SHORT"
            return "EXIT_SHORT_2BRICKS"

        return None

    def _check_loss_pause_gate(self) -> bool:
        if not self.enable_loss_pause:
            return False
        if self.loss_pause_active:
            if self.log_risk_events:
                self.write_log(
                    f"连亏暂停阻止开仓: 连续亏损批次={self.consecutive_loss_batches}, "
                    f"剩余恢复砖数={self.loss_pause_remaining_bricks}"
                )
            return True
        return False

    # ============================================================
    # 交易执行
    # ============================================================
    def _execute_open_probe(self, side: str, rf: Dict[str, Any]):
        ref_price = self.last_price_seen or self.last_brick_close or 0.0
        
        # [FIX: 资金管理模型增强]
        if self.enable_risk_based_sizing and self.margin_mode != "fixed":
            qty = self._calc_risk_based_qty(ref_price)
        else:
            standard_qty = self._calc_standard_qty(ref_price)
            open_ratio = self._get_open_ratio_by_regime(self.current_regime)
            qty = self._normalize_order_qty(standard_qty * open_ratio)

        if qty <= 0:
            return

        intent = INTENT_OPEN_LONG if side == POSITION_LONG else INTENT_OPEN_SHORT

        if self.log_decision_actions:
            self.write_log(
                f"执行开仓: side={side}, qty={qty}, ref_price={ref_price}, regime={self.current_regime}"
            )

        self.slippage_ctrl.smart_order_submit(
            intent=intent,
            volume=qty,
            action="OPEN",
            reason_code=f"OPEN_{side}",
            extra={"target_addon_stage": 0},
        )

    def _upgrade_probe_to_core(self):
        self.position_role = ROLE_CORE
        if self.log_position_events:
            self.write_cta_log("仓位角色升级: PROBE -> CORE")

    def _execute_addon(self, stage_to_enter: int, rf: Dict[str, Any]):
        ref_price = self.last_price_seen or self.last_brick_close or 0.0
        qty = self._calc_addon_order_qty(stage_to_enter, ref_price)
        if qty <= 0:
            return

        intent = INTENT_OPEN_LONG if self.pos > 0 else INTENT_OPEN_SHORT

        if self.log_decision_actions:
            self.write_log(
                f"执行加仓: stage={stage_to_enter}, dir={'LONG' if self.pos > 0 else 'SHORT'}, "
                f"qty={qty}, ref_price={ref_price}"
            )

        self.slippage_ctrl.smart_order_submit(
            intent=intent,
            volume=qty,
            action="ADDON",
            reason_code=f"ADDON_STAGE_{stage_to_enter}",
            extra={"target_addon_stage": stage_to_enter},
        )

    def _execute_reduce_addon(self, reason_code: str):
        qty = self._calc_reduce_qty()
        if qty <= 0:
            return

        # [FIX: A01] 设置减仓冷却
        self.reduce_cooldown_remaining = self.reduce_cooldown_bricks

        ref_price = self.last_price_seen or self.last_brick_close or 0.0
        intent = INTENT_CLOSE_LONG if self.pos > 0 else INTENT_CLOSE_SHORT

        if self.log_decision_actions:
            self.write_log(
                f"执行减仓: reason={reason_code}, dir={'LONG' if self.pos > 0 else 'SHORT'}, "
                f"qty={qty}, ref_price={ref_price}, 冷却{self.reduce_cooldown_bricks}砖"
            )

        self.slippage_ctrl.smart_order_submit(
            intent=intent,
            volume=qty,
            action="REDUCE",
            reason_code=reason_code,
        )

    def _execute_full_exit(self, reason_code: str):
        if self.pos == 0:
            return

        ref_price = self.last_price_seen or self.last_brick_close or 0.0
        intent = INTENT_CLOSE_LONG if self.pos > 0 else INTENT_CLOSE_SHORT

        if reason_code.startswith("HARD_REVERSAL") and self.log_risk_events:
            self.write_log(
                f"硬反转触发: {reason_code}, 冷却砖数={self.hard_reversal_cooldown_bricks}",
                level=logging.WARNING,
            )

        if self.log_decision_actions:
            self.write_log(
                f"执行平仓: reason={reason_code}, dir={'LONG' if self.pos > 0 else 'SHORT'}, "
                f"qty={abs(self.pos)}, ref_price={ref_price}"
            )

        self.slippage_ctrl.smart_order_submit(
            intent=intent,
            volume=abs(self.pos),
            action="EXIT",
            reason_code=reason_code,
        )

        if reason_code.startswith("HARD_REVERSAL"):
            self.hard_reversal_cooldown_remaining = self.hard_reversal_cooldown_bricks

    # ============================================================
    # 状态与批次管理
    # ============================================================
    def _apply_trade_to_strategy_state(self, trade: TradeData):
        filled_qty = float(trade.volume)
        fill_price = float(trade.price)
        
        # [FIX: A03+A05] 更新内部ledger
        if self.enable_ledger_fix:
            direction_str = "LONG" if trade.direction == Direction.LONG else "SHORT"
            offset_str = "OPEN" if trade.offset == Offset.OPEN else "CLOSE"
            self.ledger.apply_trade(direction_str, offset_str, fill_price, filled_qty)
            
            # 验证一致性
            if abs(self.ledger.net_qty - self.pos) > 1e-8:
                self.write_log(
                    f"警告：持仓不一致！ledger={self.ledger.net_qty}, vnpy={self.pos}",
                    level=logging.WARNING
                )
        
        order_meta = self.slippage_ctrl.order_metas.get(trade.vt_orderid, {})
        action = order_meta.get("action", "")
        target_addon_stage = order_meta.get("extra", {}).get("target_addon_stage", None)

        # ----------------------------
        # 开仓 / 加仓
        # ----------------------------
        if trade.offset == Offset.OPEN:
            if self.current_batch is None:
                direction_str = POSITION_LONG if trade.direction == Direction.LONG else POSITION_SHORT
                self.position_role = ROLE_PROBE
                self.entry_anchor_price = fill_price
                
                # [FIX: A03+A05] 设置均价
                if not self.enable_ledger_fix:
                    self.avg_entry_price = fill_price
                    
                self.current_batch = TradeBatch(
                    batch_id=f"{trade.vt_orderid}_batch_{int(time.time())}",
                    direction=direction_str,
                    entry_ts=int(time.time() * 1000),
                    entry_price=fill_price,
                    entry_count=1,
                    total_entry_qty=filled_qty,
                )

                if self.log_position_events:
                    self.write_log(
                        f"新交易批次开始: direction={direction_str}, entry_price={fill_price}, qty={filled_qty}"
                    )
            else:
                # 加仓时更新均价
                if not self.enable_ledger_fix:
                    prev_qty = self.current_batch.total_entry_qty
                    new_total_qty = prev_qty + filled_qty
                    if new_total_qty > 0:
                        self.avg_entry_price = (
                            self.avg_entry_price * prev_qty + fill_price * filled_qty
                        ) / new_total_qty

                self.current_batch.total_entry_qty += filled_qty
                self.current_batch.entry_count += 1

                if self.position_role == ROLE_PROBE:
                    self.position_role = ROLE_CORE

            # 若是 addon，推进 addon 状态
            if action == "ADDON":
                if isinstance(target_addon_stage, int):
                    self.addon_stage = max(self.addon_stage, target_addon_stage)
                else:
                    self.addon_stage += 1
                self.addon_total_qty += filled_qty
                self.last_addon_anchor_price = fill_price
                self.trend_has_advanced_since_addon = False

            self._update_unrealized_pnl_with_last_price()
            return

        # ----------------------------
        # 平仓 / 减仓
        # ----------------------------
        if trade.offset in [Offset.CLOSE, Offset.CLOSETODAY, Offset.CLOSEYESTERDAY]:
            if self.current_batch:
                self.current_batch.exit_count += 1
                self.current_batch.total_exit_qty += filled_qty
                
                # 获取当前均价
                if self.enable_ledger_fix:
                    current_avg_price = self.ledger.avg_entry_price
                else:
                    current_avg_price = self.avg_entry_price

                # 计算本次成交的已实现盈亏
                if self.current_batch.direction == POSITION_LONG:
                    realized = (fill_price - current_avg_price) * filled_qty * self.contract_size
                else:
                    realized = (current_avg_price - fill_price) * filled_qty * self.contract_size

                self.current_batch.realized_pnl += realized

            if action == "REDUCE":
                self.addon_total_qty = max(0.0, self.addon_total_qty - filled_qty)
                self.reduce_anchor_price = fill_price
                self.trend_has_advanced_since_addon = False
                if self.reset_addon_stage_after_reduce:
                    self.addon_stage = 0

            # 检查仓位是否归零
            current_pos = self.ledger.net_qty if self.enable_ledger_fix else self.pos
            if abs(current_pos) < 1e-8:
                self._close_current_batch_if_needed()
                self.position_role = ROLE_NONE
                self.addon_stage = 0
                self.addon_total_qty = 0.0
                self.entry_anchor_price = 0.0
                self.last_addon_anchor_price = 0.0
                self.reduce_anchor_price = 0.0
                self.trend_has_advanced_since_addon = False
                if self.enable_ledger_fix:
                    self.ledger.reset()
                else:
                    self.avg_entry_price = 0.0

        self._update_unrealized_pnl_with_last_price()

    def _update_unrealized_pnl_with_last_price(self):
        """[FIX: A02] 更新浮动盈亏，并检查趋势是否有效推进"""
        # 获取当前持仓和均价
        if self.enable_ledger_fix:
            current_pos = self.ledger.net_qty
            avg_price = self.ledger.avg_entry_price
        else:
            current_pos = self.pos
            avg_price = self.avg_entry_price
            
        if current_pos == 0 or self.last_price_seen is None or avg_price <= 0:
            self.unrealized_pnl = 0.0
            self.unrealized_pnl_bricks = 0.0
            return

        if current_pos > 0:
            price_diff = self.last_price_seen - avg_price
        else:
            price_diff = avg_price - self.last_price_seen

        self.unrealized_pnl_bricks = price_diff / max(self.brick_size, 1e-12)
        self.unrealized_pnl = price_diff * abs(current_pos) * self.contract_size

        # [FIX: A02] 检查趋势是否有效推进（需要超过最小阈值）
        if self.enable_reduce_logic_fix and self.addon_total_qty > 0 and self.last_addon_anchor_price > 0:
            disp = self._calc_brick_displacement_from_anchor(self.last_addon_anchor_price)
            # 只有位移超过最小阈值，才认为趋势有效推进
            if disp >= self.min_advance_for_reduce_bricks:
                self.trend_has_advanced_since_addon = True
        elif self.addon_total_qty > 0 and self.last_addon_anchor_price > 0:
            # 原有逻辑
            disp = self._calc_brick_displacement_from_anchor(self.last_addon_anchor_price)
            if disp > 0:
                self.trend_has_advanced_since_addon = True

    def _update_trade_batch_excursions(self):
        if not self.current_batch:
            return
        self.current_batch.mfe = max(self.current_batch.mfe, self.unrealized_pnl)
        self.current_batch.mae = min(self.current_batch.mae, self.unrealized_pnl)

    def _close_current_batch_if_needed(self):
        if not self.current_batch:
            return

        qty_base = max(self.current_batch.total_entry_qty, self.min_qty)
        self.current_batch.realized_pnl_bricks = (
            self.current_batch.realized_pnl
            / max(self.brick_size * self.contract_size * qty_base, 1e-12)
        )

        regime_summary: Dict[str, int] = {}
        for r in self.current_batch.regime_history:
            regime_summary[r] = regime_summary.get(r, 0) + 1

        if self.log_position_events:
            self.write_log(
                f"交易批次结束: ID={self.current_batch.batch_id}, "
                f"方向={self.current_batch.direction}, "
                f"盈亏={self.current_batch.realized_pnl:.2f}, "
                f"持有砖数={self.current_batch.holding_bricks}, "
                f"入场次数={self.current_batch.entry_count}, "
                f"离场次数={self.current_batch.exit_count}, "
                f"状态分布={regime_summary}"
            )

        self._update_loss_streak_after_batch_close(self.current_batch.realized_pnl)
        self.current_batch = None

    def _update_loss_streak_after_batch_close(self, realized_pnl: float):
        if not self.enable_loss_pause:
            return

        if realized_pnl < 0:
            self.consecutive_loss_batches += 1
        else:
            self.consecutive_loss_batches = 0

        if self.consecutive_loss_batches >= self.max_consecutive_losses:
            self.loss_pause_active = True
            self.loss_pause_remaining_bricks = self.pause_recover_bricks
            if self.log_risk_events:
                self.write_log(
                    f"连续亏损暂停触发: 次数={self.consecutive_loss_batches}, "
                    f"暂停砖数={self.loss_pause_remaining_bricks}",
                    level=logging.WARNING
                )

    # ============================================================
    # 参数/辅助函数
    # ============================================================
    def _validate_params(self):
        """[FIX: C02] 增强参数校验"""
        if self.brick_size <= 0:
            raise ValueError("brick_size must be > 0")
        if self.min_qty <= 0:
            raise ValueError("min_qty must be > 0")
        if self.qty_step <= 0:
            raise ValueError("qty_step must be > 0")
        if self.price_tick <= 0:
            raise ValueError("price_tick must be > 0")
        if self.regime_lookback_bricks < 2:
            raise ValueError("regime_lookback_bricks must be >= 2")
            
        # [FIX: C02] 新增参数校验
        if self.enable_param_validation_fix:
            # 校验比例参数范围
            ratio_params = [
                ("addon1_ratio", self.addon1_ratio, 0.0, 1.0),
                ("addon2_ratio", self.addon2_ratio, 0.0, 1.0),
                ("addon3_ratio", self.addon3_ratio, 0.0, 1.0),
                ("reduce_addon_ratio", self.reduce_addon_ratio, 0.0, 1.0),
                ("open_ratio_trend_clean", self.open_ratio_trend_clean, 0.0, 2.0),
                ("open_ratio_trend_noisy", self.open_ratio_trend_noisy, 0.0, 2.0),
                ("open_ratio_mean_revert", self.open_ratio_mean_revert, 0.0, 2.0),
                ("open_ratio_shock", self.open_ratio_shock, 0.0, 2.0),
                ("risk_ratio", self.risk_ratio, 0.0, 0.5),
            ]
            
            for name, value, min_val, max_val in ratio_params:
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
            
            # 校验触发参数
            trigger_params = [
                ("open_trigger_bricks", self.open_trigger_bricks, 1, 20),
                ("close_trigger_bricks", self.close_trigger_bricks, 1, 20),
                ("addon1_trigger_bricks", self.addon1_trigger_bricks, 1, 20),
                ("addon2_trigger_bricks", self.addon2_trigger_bricks, 1, 20),
                ("addon3_trigger_bricks", self.addon3_trigger_bricks, 1, 20),
                ("reduce_on_pullback_bricks", self.reduce_on_pullback_bricks, 1, 20),
            ]
            
            for name, value, min_val, max_val in trigger_params:
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
            
            # 校验阈值参数
            threshold_params = [
                ("trend_clean_enter_threshold", self.trend_clean_enter_threshold, 0.0, 1.0),
                ("trend_clean_exit_threshold", self.trend_clean_exit_threshold, 0.0, 1.0),
                ("hurst_trend_threshold", self.hurst_trend_threshold, 0.0, 1.0),
                ("hurst_mean_revert_threshold", self.hurst_mean_revert_threshold, 0.0, 1.0),
                ("hurst_addon_floor", self.hurst_addon_floor, 0.0, 1.0),
            ]
            
            for name, value, min_val, max_val in threshold_params:
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
            
            # 校验修复相关参数
            if self.min_advance_for_reduce_bricks < 0:
                raise ValueError(f"min_advance_for_reduce_bricks must be >= 0, got {self.min_advance_for_reduce_bricks}")
            
            if self.min_bricks_for_trading < 0:
                raise ValueError(f"min_bricks_for_trading must be >= 0, got {self.min_bricks_for_trading}")
                
            if self.account_balance <= 0:
                raise ValueError(f"account_balance must be > 0, got {self.account_balance}")

    def _safe_mean(self, arr: List[float], default: float = 0.0) -> float:
        if not arr:
            return default
        return sum(arr) / len(arr)

    def _clamp(self, x: float, low: float, high: float) -> float:
        return max(low, min(high, x))

    def _normalize_order_qty(self, qty: float) -> float:
        if qty <= 0:
            return 0.0
        steps = math.floor(qty / self.qty_step)
        norm = steps * self.qty_step
        if norm < self.min_qty:
            return 0.0
        return round(norm, 12)

    def _normalize_order_price(self, price: float, intent: str = None) -> float:
        """[FIX: C01] 价格归一化，根据订单方向处理"""
        if price <= 0:
            return 0.0
            
        if not self.enable_price_normalization_fix or intent is None:
            # 原有逻辑：四舍五入
            steps = round(price / self.price_tick)
            return round(steps * self.price_tick, 12)
        
        # 新逻辑：根据订单方向决定取整方式
        steps = price / self.price_tick
        
        if intent in [INTENT_OPEN_LONG, INTENT_CLOSE_SHORT]:
            # 买入方向：向下取整，确保不高于参考价
            steps = math.floor(steps)
        elif intent in [INTENT_OPEN_SHORT, INTENT_CLOSE_LONG]:
            # 卖出方向：向上取整，确保不低于参考价
            steps = math.ceil(steps)
        else:
            # 默认四舍五入
            steps = round(steps)
            
        return round(steps * self.price_tick, 12)

    def _should_sample_tick_log(self) -> bool:
        return True

    def _calc_standard_qty(self, ref_price: float) -> float:
        if ref_price <= 0:
            return 0.0

        if self.margin_mode == "fixed":
            notional = self.base_margin_fixed * self.default_leverage
        else:
            notional = self.base_margin_ratio * ref_price * self.default_leverage

        qty = notional / max(ref_price * self.contract_size, 1e-12)
        return self._normalize_order_qty(qty)
        
    def _calc_risk_based_qty(self, ref_price: float) -> float:
        """[FIX: 资金管理模型增强] 基于风险的头寸计算"""
        if ref_price <= 0 or not self.enable_risk_based_sizing:
            return self._calc_standard_qty(ref_price)
        
        # 1. 计算本笔交易愿意承担的最大风险金额
        risk_capital = self.account_balance * self.risk_ratio
        
        # 2. 计算止损距离（砖块数转价格）
        stop_loss_price_distance = self.stop_loss_bricks * self.brick_size
        
        # 3. 计算应交易的标的数量
        qty_asset = risk_capital / stop_loss_price_distance
        
        # 4. 转换为合约张数
        qty_contract = qty_asset / self.contract_face_value
        
        # 5. 应用合约的最小交易单位限制
        qty_contract_normalized = self._normalize_order_qty(qty_contract)
        
        # 6. 与传统方法比较，取较小值
        traditional_qty = self._calc_standard_qty(ref_price)
        
        return min(qty_contract_normalized, traditional_qty)

    def _get_open_ratio_by_regime(self, regime: str) -> float:
        if regime == REGIME_TREND_CLEAN:
            return self.open_ratio_trend_clean
        if regime == REGIME_TREND_NOISY:
            return self.open_ratio_trend_noisy
        if regime == REGIME_MEAN_REVERT:
            return self.open_ratio_mean_revert
        if regime == REGIME_SHOCK:
            return self.open_ratio_shock
        return self.open_ratio_mean_revert

    def _get_max_addons_by_regime(self, regime: str) -> int:
        if regime == REGIME_TREND_CLEAN:
            return self.max_addons_trend_clean
        if regime == REGIME_TREND_NOISY:
            return self.max_addons_trend_noisy
        if regime == REGIME_MEAN_REVERT:
            return self.max_addons_mean_revert
        if regime == REGIME_SHOCK:
            return self.max_addons_shock
        return 0

    def _regime_allows_addon(self, regime: str, hurst_value: float) -> bool:
        if regime not in [REGIME_TREND_CLEAN, REGIME_TREND_NOISY]:
            return False
        if hurst_value < self.hurst_addon_floor:
            return False
        return True

    def _get_required_same_dir_bricks_for_addon_stage(self, stage: int) -> int:
        if stage == 1:
            return self.addon1_trigger_bricks
        if stage == 2:
            return self.addon2_trigger_bricks
        if stage == 3:
            return self.addon3_trigger_bricks
        return 999

    def _get_addon_reference_anchor(self) -> float:
        if self.last_addon_anchor_price > 0:
            return self.last_addon_anchor_price
        if self.reduce_anchor_price > 0:
            return self.reduce_anchor_price
        return self.entry_anchor_price

    def _calc_brick_displacement_from_anchor(self, anchor_price: float) -> float:
        if anchor_price <= 0:
            return 0.0

        ref = self.last_brick_close or self.last_price_seen or 0.0
        if ref <= 0:
            return 0.0

        if self.pos > 0:
            return max((ref - anchor_price) / self.brick_size, 0.0)
        elif self.pos < 0:
            return max((anchor_price - ref) / self.brick_size, 0.0)
        else:
            # 空仓时用于某些辅助判断，按最近方向近似
            if self.current_run_dir > 0:
                return max((ref - anchor_price) / self.brick_size, 0.0)
            elif self.current_run_dir < 0:
                return max((anchor_price - ref) / self.brick_size, 0.0)
            return 0.0

    def _calc_addon_order_qty(self, stage_to_enter: int, ref_price: float) -> float:
        # [FIX: 资金管理模型增强]
        if self.enable_risk_based_sizing and self.margin_mode != "fixed":
            base_qty = self._calc_risk_based_qty(ref_price)
        else:
            base_qty = self._calc_standard_qty(ref_price)
            
        if base_qty <= 0:
            return 0.0

        if stage_to_enter == 1:
            ratio = self.addon1_ratio
        elif stage_to_enter == 2:
            ratio = self.addon2_ratio
        elif stage_to_enter == 3:
            ratio = self.addon3_ratio
        else:
            return 0.0

        return self._normalize_order_qty(base_qty * ratio)

    def _calc_reduce_qty(self) -> float:
        if self.addon_total_qty <= 0:
            return 0.0
        qty = self.addon_total_qty * self.reduce_addon_ratio
        qty = min(qty, abs(self.pos))
        return self._normalize_order_qty(qty)

    def write_cta_log(self, msg: str):
        self.write_log(msg)

    # ============================================================
    # 可选：状态输出
    # ============================================================
    def on_timer(self):
        """[FIX: C07] 统一扫描入口，避免重复扫描"""
        # 如果启用了重复扫描修复，只在on_one_sec_bar中扫描
        if not self.enable_duplicate_scan_fix:
            self._scan_active_orders_for_timeout()
            self._process_pending_fallbacks()