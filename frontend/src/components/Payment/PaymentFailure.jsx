import React, { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

export default function PaymentFailure() {
  const navigate = useNavigate();
  const [params] = useSearchParams();

  const txnid = params.get("txnid");
  const amount = params.get("amount");

  useEffect(() => {
    setTimeout(() => navigate("/"), 3000);
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-red-50 text-center">
      <h1 className="text-4xl font-bold text-red-700">Payment Failed ❌</h1>
      <p className="text-lg mt-4">Transaction ID: {txnid}</p>
      <p className="text-lg">Amount Attempted: ₹{amount}</p>
      <p className="mt-6 text-gray-600">Redirecting to homepage...</p>
    </div>
  );
}
