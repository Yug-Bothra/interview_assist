import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import requests
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Supabase client
from supabase import create_client, Client

load_dotenv()

router = APIRouter(prefix="/payment", tags=["payment"])

# -------------------------------------------------------
# PAYU CONFIG
# -------------------------------------------------------

PAYU_KEY = os.getenv("PAYU_KEY")
PAYU_SALT = os.getenv("PAYU_SALT")

# Default to TEST environment if variable missing
PAYU_BASE_URL = os.getenv("PAYU_BASE_URL") or "https://test.payu.in/_payment"

PAYU_SURL = os.getenv("PAYU_SURL", "https://interview-assist-1.onrender.com/payment/payu_success")
PAYU_FURL = os.getenv("PAYU_FURL", "https://interview-assist-1.onrender.com/payment/payu_failure")

# -------------------------------------------------------
# SUPABASE CONFIG
# -------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# -------------------------------------------------------
# INR → USD Conversion
# -------------------------------------------------------

def convert_inr_to_usd(amount_in_inr: float) -> float:
    try:
        res = requests.get(
            "https://api.exchangerate.host/convert?from=INR&to=USD",
            timeout=5
        ).json()
        rate = res.get("result")
        return round(amount_in_inr * (rate or 1 / 83), 2)
    except:
        return round(amount_in_inr / 83, 2)

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def generate_txnid() -> str:
    return secrets.token_hex(12)

def generate_request_hash(params: Dict[str, str]) -> str:
    seq = [
        PAYU_KEY,
        params.get("txnid", ""),
        params.get("amount", ""),
        params.get("productinfo", ""),
        params.get("firstname", ""),
        params.get("email", "")
    ]

    # udf1–udf10
    for i in range(1, 11):
        seq.append(params.get(f"udf{i}", ""))

    seq.append(PAYU_SALT)

    return hashlib.sha512("|".join(seq).encode()).hexdigest().lower()

def verify_response_hash(posted: Dict[str, str]) -> bool:
    received_hash = posted.get("hash", "")

    seq = [PAYU_SALT, posted.get("status", "")]

    for i in range(10, 0, -1):
        seq.append(posted.get(f"udf{i}", ""))

    seq.extend([
        posted.get("email", ""),
        posted.get("firstname", ""),
        posted.get("productinfo", ""),
        posted.get("amount", ""),
        posted.get("txnid", ""),
        PAYU_KEY
    ])

    calc = hashlib.sha512("|".join(seq).encode()).hexdigest().lower()
    return calc == received_hash.lower()

def compute_subscription_dates(billing_period: str):
    start = datetime.utcnow()
    days = {"monthly": 30, "quarterly": 90, "yearly": 365}.get(billing_period, 30)
    return start, start + timedelta(days=days)

# -------------------------------------------------------
# PAYMENT REQUEST BODY
# -------------------------------------------------------

class CreatePaymentRequest(BaseModel):
    user_id: str
    plan: str
    billing_period: str
    firstname: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    amount: str

# -------------------------------------------------------
# CREATE PAYMENT REQUEST
# -------------------------------------------------------

@router.post("/create")
async def create_payment(req: CreatePaymentRequest):

    if not PAYU_KEY or not PAYU_SALT:
        raise HTTPException(status_code=500, detail="PayU is not configured")

    txnid = generate_txnid()
    amount_str = f"{float(req.amount):.2f}"

    params = {
        "key": PAYU_KEY,
        "txnid": txnid,
        "amount": amount_str,
        "productinfo": f"{req.plan} subscription ({req.billing_period})",
        "firstname": req.firstname or "User",
        "email": req.email or "user@example.com",
        "phone": req.phone or "",
        "surl": PAYU_SURL,
        "furl": PAYU_FURL,
        "udf1": req.user_id
    }

    # Insert pending payment
    if supabase:
        supabase.table("payments").insert({
            "txnid": txnid,
            "user_id": req.user_id,
            "plan": req.plan,
            "billing_period": req.billing_period,
            "amount_in_inr": float(req.amount),
            "amount_in_usd": convert_inr_to_usd(float(req.amount)),
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }).execute()

    params["hash"] = generate_request_hash(params)

    # Build auto-submit HTML
    form_inputs = "\n".join(
        f'<input type="hidden" name="{k}" value="{v}" />'
        for k, v in params.items()
    )

    form_html = f"""
    <html>
    <body onload="document.forms[0].submit();">
        <form method="post" action="{PAYU_BASE_URL}">
            {form_inputs}
        </form>
    </body>
    </html>
    """

    return JSONResponse({"form": form_html, "txnid": txnid})

# -------------------------------------------------------
# PAYU SUCCESS CALLBACK
# -------------------------------------------------------

@router.post("/payu_success")
async def payu_success(request: Request):
    posted = dict(await request.form())

    txnid = posted.get("txnid")
    amount = float(posted.get("amount", 0))
    verified = verify_response_hash(posted)

    status = "success" if verified else "failed"

    # Update payment status
    if supabase:
        supabase.table("payments").update({
            "status": status,
            "amount_in_inr": amount,
            "amount_in_usd": convert_inr_to_usd(amount),
            "response": posted,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("txnid", txnid).execute()

    # Activate subscription
    if verified:
        productinfo = posted.get("productinfo", "")
        plan = productinfo.split(" ")[0]
        billing = productinfo.split("(")[1].split(")")[0]

        start, end = compute_subscription_dates(billing)

        supabase.table("users").update({
            "subscription_tier": plan,
            "subscription_status": "active",
            "subscription_start_date": start.isoformat(),
            "subscription_end_date": end.isoformat()
        }).eq("id", posted.get("udf1")).execute()

    return HTMLResponse(f"""
        <h2>Payment Success</h2>
        <p>Transaction ID: {txnid}</p>
        <p>Amount: ₹{amount}</p>
        <a href="/">Return to App</a>
    """)

# -------------------------------------------------------
# PAYU FAILURE CALLBACK
# -------------------------------------------------------

@router.post("/payu_failure")
async def payu_failure(request: Request):
    posted = dict(await request.form())
    txnid = posted.get("txnid")

    if supabase:
        supabase.table("payments").update({
            "status": "failed",
            "response": posted,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("txnid", txnid).execute()

    return HTMLResponse(f"""
        <h2>Payment Failed</h2>
        <p>Transaction ID: {txnid}</p>
        <a href="/">Return to App</a>
    """)

def get_router():
    return router
