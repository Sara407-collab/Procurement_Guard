"""
Master data: vendors and employees.

The four fields that matter most here are bank_account, address, phone and
tax_id. Nothing in a transaction table can tell you a vendor and a buyer share
a bank account — that link only exists *between* records. It is exactly the
relationship a graph finds and a tabular model never will.
"""

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------- name pieces
_V_PREFIX = ["Apex", "Meridian", "Crescent", "Summit", "Ironwood", "Blue Harbor",
             "Northgate", "Silverline", "Pinnacle", "Vertex", "Cobalt", "Granite",
             "Riverstone", "Falcon", "Onyx", "Trident", "Cedarpoint", "Halcyon"]
_V_MID = ["Trading", "Industrial", "Technologies", "Supplies", "Engineering",
          "Logistics", "Solutions", "Enterprises", "Systems", "Contractors"]
_V_SUFFIX = ["Ltd", "Pvt Ltd", "LLC", "Inc", "& Co", "Group", "Corp"]

_FIRST = ["Ayesha", "Bilal", "Daniel", "Elena", "Farhan", "Grace", "Hassan",
          "Imran", "Julia", "Kamran", "Laila", "Marcus", "Nadia", "Omar",
          "Priya", "Rashid", "Sana", "Tariq", "Usman", "Vera", "Wasim", "Zara"]
_LAST = ["Ahmed", "Baig", "Chowdhury", "Dawson", "Ellis", "Farooq", "Gill",
         "Hussain", "Iqbal", "Javed", "Khan", "Lopez", "Malik", "Nawaz",
         "Osei", "Patel", "Qureshi", "Raza", "Siddiqui", "Tanaka"]

_STREETS = ["Mall Road", "Ferozepur Road", "Gulberg Ave", "Canal Bank Rd",
            "Jail Road", "Model Town Link", "Raiwind Rd", "Multan Rd",
            "Shahrah-e-Quaid", "Davis Road", "Empress Road", "Cavalry Ground"]
_CITIES = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Sialkot"]


def _bank_account(rng):
    return f"PK{rng.integers(10, 99)}TMCB{rng.integers(10**10, 10**11 - 1)}"


def _phone(rng):
    return f"+92-3{rng.integers(0, 10)}{rng.integers(0, 10)}-{rng.integers(1000000, 9999999)}"


def _address(rng):
    return f"{rng.integers(1, 400)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"


def _tax_id(rng):
    return f"{rng.integers(1000000, 9999999)}-{rng.integers(0, 10)}"


def build_vendors(rng: np.random.Generator) -> pd.DataFrame:
    """Legitimate vendor master. Fraud is layered on later, never here."""
    rows, seen = [], set()
    start = pd.Timestamp("2015-01-01")
    span_days = (pd.Timestamp("2024-01-01") - start).days

    for i in range(C.N_VENDORS):
        # Occasionally produce near-duplicate names on purpose. Real vendor
        # masters are full of them, and entity resolution is a feature we want.
        while True:
            name = (f"{rng.choice(_V_PREFIX)} {rng.choice(_V_MID)} "
                    f"{rng.choice(_V_SUFFIX)}")
            if name not in seen:
                seen.add(name)
                break

        rows.append({
            "vendor_id":         f"V{i:04d}",
            "vendor_name":       name,
            "tax_id":            _tax_id(rng),
            "bank_account":      _bank_account(rng),
            "address":           _address(rng),
            "phone":             _phone(rng),
            "registration_date": start + pd.Timedelta(days=int(rng.integers(0, span_days))),
            "primary_category":  rng.choice(C.CATEGORIES),
            "is_ghost":          False,   # ground truth, set by fraud_injection
        })

    return pd.DataFrame(rows)


def build_employees(rng: np.random.Generator) -> pd.DataFrame:
    """Buyers / approvers. These are the people who can collude with a vendor."""
    rows = []
    start = pd.Timestamp("2012-01-01")
    span_days = (pd.Timestamp("2023-06-01") - start).days

    for i in range(C.N_EMPLOYEES):
        rows.append({
            "employee_id":   f"E{i:04d}",
            "employee_name": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
            "department":    rng.choice(C.DEPARTMENTS),
            "tax_id":        _tax_id(rng),
            "bank_account":  _bank_account(rng),
            "address":       _address(rng),
            "phone":         _phone(rng),
            "hire_date":     start + pd.Timedelta(days=int(rng.integers(0, span_days))),
            "approval_limit": int(rng.choice(C.APPROVAL_THRESHOLDS)),
        })

    return pd.DataFrame(rows)
