# =====================
# Build Stage
# =====================
FROM gcc:13 AS builder

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y cmake

RUN mkdir build && cd build && \
    cmake .. && \
    cmake --build .

# =====================
# Runtime Stage (same GCC to avoid mismatch)
# =====================
FROM gcc:13

WORKDIR /app
COPY --from=builder /app/build/eigen_engine .

CMD ["./eigen_engine"]