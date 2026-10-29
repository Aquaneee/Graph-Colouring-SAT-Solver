# Exact SAT Solving for the Graph Coloring Problem (GCP)

Repository cài đặt và thực nghiệm Bài tập lớn số 2 - Môn Các vấn đề hiện đại trong KHMT.

## Thành viên thực hiện

* **Họ và tên:** Nguyễn Đức Mạnh, Hà Phú Duy, Đặng Nhật Quang
* **Trường:** Đại học Công nghệ – ĐHQGHN (VNU-UET)

## Giới thiệu

Dự án tập trung giải quyết bài toán Tô màu đồ thị (Graph Coloring Problem - GCP):

1. Cài đặt 5 kỹ thuật At-Most-One (AMO): Pairwise, Sequential Counter, Binary, Commander, Product Encoding.
2. Mô hình ghép cặp đỉnh theo cây Zykov (Zykov's Tree) với cơ chế lặp tăng trưởng CEGAR Incremental SAT (CP 2019).
3. Đánh giá đối đầu với các công cụ giải chính xác công nghiệp: Google OR-Tools CP-SAT, Gurobi (ILP), CPLEX.

## Hướng dẫn cài đặt & Chạy thực nghiệm

```bash

# 1. Cài đặt các thư viện phụ thuộc

pip install -r requirements.txt

# 2. Chạy pipeline thực nghiệm

python gcp_benchmark_[runner.py](http://runner.py) --data-dir data --output-dir results --amo-timeout 100 --solver-timeout 300
```