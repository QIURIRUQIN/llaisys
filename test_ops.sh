# Optional: [add, argmax, embedding, linear, rms_norm, rope, self_attention, swiglu]
ENABLE_TRITON=True \
    python test/ops/self_attention.py --device nvidia
