import React, { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

export default function PaymentSuccess() {
  const navigate = useNavigate();
  const [params] = useSearchParams();

  const txnid = params.get("txnid");
  const amount = params.get("amount");

  useEffect(() => {
    const timer = setTimeout(() => navigate("/"), 3000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-green-50 text-center px-6">
      <h1 className="text-4xl font-bold text-green-700">Payment Successful 🎉</h1>

      <div className="mt-6 bg-white shadow-md rounded-xl px-6 py-4 w-full max-w-md">
        <p className="text-lg text-gray-700">
          <strong>Transaction ID:</strong> {txnid || "N/A"}
        </p>
        <p className="text-lg text-gray-700">
          <strong>Amount Paid:</strong> ₹{amount || "0"}
        </p>
      </div>

      <p className="mt-8 text-gray-600 text-sm">
        Redirecting to homepage...
      </p>
    </div>
  );
}
