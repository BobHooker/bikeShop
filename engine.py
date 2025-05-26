"""
Core simulation logic for the BikeSim GPT demo.
Gross margin ≈ 20 %; overhead, recalls, credit, popularity feedback included.
"""

from __future__ import annotations
import datetime, random, numpy as np
from dataclasses import dataclass, field
import pandas as pd

# ────────────────────────────────────────────────────
# Simulation-wide constants (tweak here as desired)
# ────────────────────────────────────────────────────
STEP_DAYS               = 14           # one “turn” = 2 weeks
ELASTICITY              = 1.25
REF_PRICE               = 520
OVERHEAD                = 1_500        # rent, wages per period
AD_MIN, AD_MAX          = 20, 150
CREDIT_LIMIT            = -2_000
INTEREST_RATE           = 0.05         # on negative balances
CREDIT_RESET_EVERY      = 26           # periods (= 1 year)
ALPHA, DECAY            = 0.5, 0.95    # popularity feedback
POP_MIN, POP_MAX        = 0.2, 3.0

EVENT_PROB              = 5/26         # ≈ 5 supply shocks / year
MAX_SHOPS_EVENT         = 3
RECALL_DURATION         = 3            # periods

START_STOCK = {
    "Tom's Bikes":       120,
    "Bob's Wheels":      100,
    "University Cycles": 140,
    "Industrial Ride":    90,
    "Uptown Pedals":     110,
}

# ────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────
def season_factor(name: str, d: datetime.date) -> float:
    if name == "University Cycles":
        return 0.1 if d.month in (6, 7, 8) else 1.0
    if d.month == 12 and 11 <= d.day <= 24:
        return 3.0
    return 0.6 if d.month in (10, 11, 12, 1, 2, 3, 4, 5) else 1.0

def wholesale_ratio(pop: float) -> float:
    """Lower = cheaper. 0.78-0.83 gives ~20 % margin."""
    return min(0.83, 0.78 + 0.018 * (pop - 1))

def demand_mu(base, ad, price, season, pop) -> float:
    return base * pop * season * (1 + ad/500) * (REF_PRICE / price) ** ELASTICITY

# ────────────────────────────────────────────────────
@dataclass
class Shop:
    name: str
    price: float
    base: int
    ad: int
    capital: float
    stock: int = field(init=False)
    pop: float = 1.0
    recall: int = 0
    credit_used: int = 0       # once per year

    def __post_init__(self):
        self.stock = START_STOCK[self.name]

# ────────────────────────────────────────────────────
def run(periods: int = 52,
        *,
        seed: int | None = None,
        start_date: datetime.date | None = None) -> pd.DataFrame:
    """
    Run the simulation and return a DataFrame with results.

    Parameters
    ----------
    periods : int
        Number of bi-weekly turns (default 52 = two years).
    seed : int | None
        RNG seed for reproducibility.
    start_date : date | None
        Calendar start; default 24 May 2025 (matches original prompt).
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    if start_date is None:
        start_date = datetime.date(2025, 5, 24)

    shops: list[Shop] = [
        Shop("Tom's Bikes",       310, 20, 50, 12_500),
        Shop("Bob's Wheels",      600, 15, 70, 13_000),
        Shop("University Cycles", 450, 25, 60, 14_000),
        Shop("Industrial Ride",   280, 18, 40, 11_000),
        Shop("Uptown Pedals",     520, 17, 65, 12_000),
    ]

    records: list[dict] = []

    for period in range(periods):
        # Reset annual credit usage
        if period % CREDIT_RESET_EVERY == 0:
            for sh in shops: sh.credit_used = 0

        date = start_date + datetime.timedelta(days=STEP_DAYS * period)

        # Random supply-chain shock
        if random.random() < EVENT_PROB and shops:
            for v in random.sample(shops, min(MAX_SHOPS_EVENT, len(shops))):
                v.recall = RECALL_DURATION

        bankrupt: list[Shop] = []

        for sh in shops:
            season = season_factor(sh.name, date)
            wr     = wholesale_ratio(sh.pop)

            # 1️⃣  SALES / RECALL
            if sh.recall:
                sold = 0
                profit = -sh.ad - OVERHEAD
                sh.recall -= 1
            else:
                mu = demand_mu(sh.base, sh.ad, sh.price, season, sh.pop)
                sold = min(np.random.poisson(mu), sh.stock)
                profit = sold * sh.price - sh.ad - OVERHEAD
                sh.stock -= sold

            # 2️⃣  RE-ORDER (if not in recall)
            if sh.recall == 0:
                need = START_STOCK[sh.name] - sh.stock
                if need > 0:
                    unit = sh.price * wr
                    budget = max(0, sh.capital - CREDIT_LIMIT)
                    order  = min(need, int(budget // unit))
                    sh.stock   += order
                    sh.capital -= order * unit

            # 3️⃣  CAPITAL & CREDIT
            sh.capital += profit
            if sh.capital < 0:  # interest
                sh.capital *= (1 + INTEREST_RATE)

            if sh.capital < 0 and sh.credit_used == 0 and sh.capital > CREDIT_LIMIT:
                sh.credit_used = 1                 # take lifeline once
            elif sh.capital <= CREDIT_LIMIT:
                bankrupt.append(sh)

            # 4️⃣  POPULARITY
            pct = sold / START_STOCK[sh.name]
            sh.pop = max(POP_MIN, min(POP_MAX, sh.pop * DECAY + ALPHA * pct))

            # 5️⃣  Record row
            records.append({
                "Date":    date,
                "Period":  period+1,
                "Store":   sh.name,
                "RC":      sh.recall,
                "Price":   sh.price,
                "Ad":      sh.ad,
                "Pop":     round(sh.pop,2),
                "Stock":   sh.stock,
                "Sold":    sold,
                "Profit":  round(profit,2),
                "Capital": round(sh.capital,2),
            })

        # Remove bankrupt shops
        for dead in bankrupt:
            shops.remove(dead)

        if not shops:
            break  # everyone bust

    return pd.DataFrame(records)
