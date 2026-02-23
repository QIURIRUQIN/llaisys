# ENABLE_TRITON=True \
# nsys profile \
#   --trace=cuda,nvtx,osrt \
#   --sample=none \
#   --cpuctxsw=none \
#   -o test_python_impro_3 \
#   python test/test_infer.py --test --device nvidia

ENABLE_TRITON=True \
  python test/test_infer.py --test --device nvidia
