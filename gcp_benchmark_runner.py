import os
import sys
import time
import math
import argparse
import urllib.request
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Thử import cnsplots theo yêu cầu đề bài (fallback về matplotlib nếu chưa cài)
try:
    import cnsplots as cns
    HAS_CNSPLOTS = True
except ImportError:
    HAS_CNSPLOTS = False

# Import PySAT
try:
    from pysat.solvers import Solver
    from pysat.card import CardEnc, EncType
    from pysat.formula import IDPool
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False
    print("[WARNING] Chưa cài đặt thư viện `python-sat` (PySAT). Hãy chạy: pip install python-sat")

# Import OR-Tools CP-SAT
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

# Import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

# Import CPLEX (docplex)
try:
    from docplex.mp.model import Model as CplexMIPModel
    from docplex.cp.model import CpoModel as CplexCPModel
    HAS_CPLEX = True
except ImportError:
    HAS_CPLEX = False


# ==============================================================================
# PHẦN 1: ĐỌC DỮ LIỆU ĐỒ THỊ DIMACS (.col) & TỰ ĐỘNG TẢI BENCHMARK
# ==============================================================================

class Graph:
    """Biểu diễn đồ thị vô hướng G = (V, E) với các đỉnh đánh số từ 1 đến n."""
    def __init__(self, name: str, n: int, edges: List[Tuple[int, int]], known_chi: Optional[int] = None):
        self.name = name
        self.n = n
        self.edges: Set[Tuple[int, int]] = set()
        self.adj: Dict[int, Set[int]] = {u: set() for u in range(1, n + 1)}
        for u, v in edges:
            if u != v and 1 <= u <= n and 1 <= v <= n:
                a, b = (u, v) if u < v else (v, u)
                self.edges.add((a, b))
                self.adj[a].add(b)
                self.adj[b].add(a)
        self.m = len(self.edges)
        self.known_chi = known_chi

    def is_edge(self, u: int, v: int) -> bool:
        if u > v:
            u, v = v, u
        return (u, v) in self.edges


def read_dimacs_col(filepath: str, known_chi: Optional[int] = None) -> Graph:
    """Đọc file định dạng chuẩn DIMACS .col."""
    name = os.path.splitext(os.path.basename(filepath))[0]
    n = 0
    edges = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            parts = line.split()
            if parts[0] == 'p':
                # p edge <n> <m> hoặc p col <n> <m>
                n = int(parts[2])
            elif parts[0] == 'e':
                u, v = int(parts[1]), int(parts[2])
                edges.append((u, v))
    return Graph(name, n, edges, known_chi=known_chi)


def generate_mycielski_graph(k: int) -> Graph:
    """Sinh đồ thị Mycielski M_k (với sắc số chuẩn xác chi(M_k) = k)."""
    # M_2 là đồ thị K_2 (2 đỉnh, 1 cạnh)
    n = 2
    edges = [(1, 2)]
    for step in range(3, k + 1):
        new_n = 2 * n + 1
        new_edges = list(edges)
        # Các đỉnh u_i: từ n+1 đến 2n; đỉnh w: 2n+1
        for u, v in edges:
            new_edges.append((u, n + v))
            new_edges.append((v, n + u))
        for i in range(1, n + 1):
            new_edges.append((n + i, 2 * n + 1))
        n = new_n
        edges = new_edges
    return Graph(f"myciel{k}", n, edges, known_chi=k)


def generate_queen_graph(side: int, known_chi: Optional[int] = None) -> Graph:
    """Sinh đồ thị Queen (side x side): mỗi ô là 1 đỉnh, 2 ô có cạnh nếu cùng hàng/cột/chéo."""
    n = side * side
    edges = []
    for r1 in range(side):
        for c1 in range(side):
            u = r1 * side + c1 + 1
            for r2 in range(r1, side):
                for c2 in range(side):
                    v = r2 * side + c2 + 1
                    if u >= v:
                        continue
                    if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                        edges.append((u, v))
    return Graph(f"queen{side}_{side}", n, edges, known_chi=known_chi)


def generate_kneser_or_random_graph(name: str, n: int, p: float, seed: int, known_chi: Optional[int] = None) -> Graph:
    """Sinh đồ thị ngẫu nhiên Erdos-Renyi G(n, p) có cố định seed phục vụ test nhanh."""
    rng = np.random.default_rng(seed)
    edges = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if rng.random() < p:
                edges.append((u, v))
    return Graph(name, n, edges, known_chi=known_chi)


def load_benchmark_suite(data_dir: str = "data") -> List[Graph]:
    """
    Tải các file .col từ thư mục `data_dir` nếu có sẵn.
    Nếu thư mục trống, tự động khởi tạo bộ đồ thị chuẩn (myciel, queen, random/structured)
    để chạy thực nghiệm ngay lập tức.
    """
    os.makedirs(data_dir, exist_ok=True)
    col_files = [f for f in os.listdir(data_dir) if f.endswith('.col')]
    if col_files:
        print(f"[INFO] Tìm thấy {len(col_files)} file .col trong thư mục '{data_dir}'. Đang nạp...")
        graphs = []
        for fname in sorted(col_files):
            g = read_dimacs_col(os.path.join(data_dir, fname))
            graphs.append(g)
        return graphs

    print("[INFO] Không tìm thấy file .col cục bộ. Đang khởi tạo bộ dữ liệu benchmark chuẩn...")
    graphs = [
        generate_mycielski_graph(3),                # myciel3: |V|=11, |E|=20, chi=4
        generate_mycielski_graph(4),                # myciel4: |V|=23, |E|=71, chi=5
        generate_queen_graph(5, known_chi=5),       # queen5_5: |V|=25, |E|=160, chi=5
        generate_queen_graph(6, known_chi=7),       # queen6_6: |V|=36, |E|=290, chi=7
        generate_mycielski_graph(5),                # myciel5: |V|=47, |E|=236, chi=6
        generate_queen_graph(7, known_chi=7),       # queen7_7: |V|=49, |E|=476, chi=7
        generate_kneser_or_random_graph("DSJC50.5", 50, 0.45, seed=42),
    ]
    return graphs


# ==============================================================================
# PHẦN 2: THUẬT TOÁN DSATUR (UPPER BOUND) VÀ CLIQUE (LOWER BOUND)
# ==============================================================================

def dsatur_coloring(graph: Graph) -> Tuple[int, Dict[int, int]]:
    """
    Thuật toán tham lam DSatur (Brélaz, 1979) trả về (số màu cận trên UB, phép gán màu).
    """
    colors: Dict[int, int] = {}
    neighbor_colors: Dict[int, Set[int]] = {u: set() for u in range(1, graph.n + 1)}
    degrees = {u: len(graph.adj[u]) for u in range(1, graph.n + 1)}
    uncolored = set(range(1, graph.n + 1))

    while uncolored:
        # Chọn đỉnh có bậc bão hòa (saturation degree) lớn nhất, ưu tiên bậc gốc lớn nhất
        best_u = max(uncolored, key=lambda u: (len(neighbor_colors[u]), degrees[u], -u))
        # Tìm màu nhỏ nhất hợp lệ (bắt đầu từ 1)
        c = 1
        while c in neighbor_colors[best_u]:
            c += 1
        colors[best_u] = c
        uncolored.remove(best_u)
        for v in graph.adj[best_u]:
            if v in uncolored:
                neighbor_colors[v].add(c)

    num_colors = max(colors.values()) if colors else 0
    return num_colors, colors


def greedy_clique_lb(graph: Graph) -> int:
    """Tìm một Clique cực đại theo heuristic bậc lớn nhất để làm cận dưới khởi tạo (LB)."""
    if graph.m == 0:
        return 1 if graph.n > 0 else 0
    best_clique: List[int] = []
    nodes_sorted = sorted(range(1, graph.n + 1), key=lambda u: len(graph.adj[u]), reverse=True)
    for start_node in nodes_sorted[:min(15, graph.n)]:
        clique = [start_node]
        candidates = sorted(graph.adj[start_node], key=lambda u: len(graph.adj[u]), reverse=True)
        for u in candidates:
            if all(u in graph.adj[w] for w in clique):
                clique.append(u)
        if len(clique) > len(best_clique):
            best_clique = clique
    return max(2, len(best_clique))


# ==============================================================================
# PHẦN 3: MÃ HÓA SAT TRỰC TIẾP VỚI 5 KỸ THUẬT AT-MOST-ONE (AMO)
# ==============================================================================

def build_amo_clauses(lits: List[int], amo_type: str, vpool: IDPool) -> List[List[int]]:
    """
    Sinh các mệnh đề At-Most-One (AMO) cho danh sách biến `lits` theo đúng 5 kỹ thuật trong báo cáo:
    1. 'pairwise'   : Pairwise / Binomial O(k^2) mệnh đề, 0 biến phụ.
    2. 'seqcounter' : Sequential Counter (Sinz, 2005) O(k) mệnh đề, O(k) biến phụ.
    3. 'binary'     : Binary / Bitwise Encoding O(k log k) mệnh đề, O(log k) biến phụ.
    4. 'commander'  : Commander Encoding chia nhóm kích thước g=3.
    5. 'product'    : Product Encoding (Greshuk et al.) lưới 2D p x q, O(sqrt(k)) biến phụ.
    """
    m = len(lits)
    if m <= 1:
        return []

    amo_type = amo_type.lower()

    if amo_type == 'pairwise':
        return [[-lits[i], -lits[j]] for i in range(m) for j in range(i + 1, m)]

    elif amo_type == 'seqcounter':
        cnf = CardEnc.atmost(lits=lits, bound=1, encoding=EncType.seqcounter, vpool=vpool)
        return cnf.clauses

    elif amo_type == 'binary':
        num_bits = max(1, math.ceil(math.log2(m)))
        bits = [vpool.id(f"bin_{ id(lits) }_{b}") for b in range(num_bits)]
        clauses = []
        for idx, x in enumerate(lits):
            for b in range(num_bits):
                if (idx >> b) & 1:
                    clauses.append([-x, bits[b]])
                else:
                    clauses.append([-x, -bits[b]])
        return clauses

    elif amo_type == 'commander':
        group_size = 3
        if m <= group_size:
            return [[-lits[i], -lits[j]] for i in range(m) for j in range(i + 1, m)]
        clauses = []
        commanders = []
        for g_idx in range(0, m, group_size):
            group = lits[g_idx:g_idx + group_size]
            c_var = vpool.id(f"cmd_{ id(lits) }_{g_idx}")
            commanders.append(c_var)
            # Pairwise trong nhóm
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    clauses.append([-group[i], -group[j]])
                # Nếu x_i = 1 thì commander c_var = 1
                clauses.append([-group[i], c_var])
            # Nếu c_var = 1 thì ít nhất 1 biến trong nhóm = 1
            clauses.append([-c_var] + group)
        # Đệ quy AMO trên các biến chỉ huy
        clauses.extend(build_amo_clauses(commanders, 'commander', vpool))
        return clauses

    elif amo_type == 'product':
        p = math.ceil(math.sqrt(m))
        q = math.ceil(m / p)
        row_vars = [vpool.id(f"prod_r_{ id(lits) }_{r}") for r in range(p)]
        col_vars = [vpool.id(f"prod_c_{ id(lits) }_{c}") for c in range(q)]
        clauses = []
        for idx, x in enumerate(lits):
            r = idx // q
            c = idx % q
            clauses.append([-x, row_vars[r]])
            clauses.append([-x, col_vars[c]])
        # Pairwise trên các biến hàng và biến cột
        for i in range(p):
            for j in range(i + 1, p):
                clauses.append([-row_vars[i], -row_vars[j]])
        for i in range(q):
            for j in range(i + 1, q):
                clauses.append([-col_vars[i], -col_vars[j]])
        return clauses

    else:
        raise ValueError(f"Không hỗ trợ kiểu mã hóa AMO: {amo_type}")


def encode_direct_k_coloring(graph: Graph, k: int, amo_type: str = 'product') -> Tuple[List[List[int]], int, IDPool]:
    """
    Sinh công thức SAT CNF cho bài toán k-Coloring trực tiếp với kỹ thuật AMO được chọn.
    Trả về: (danh sách mệnh đề, tổng số biến, vpool).
    """
    vpool = IDPool()
    clauses: List[List[int]] = []

    # Biến x_{v, i} cho v in 1..n, i in 1..k
    def x_var(v: int, i: int) -> int:
        return vpool.id(f"x_{v}_{i}")

    # Khởi tạo ID cho toàn bộ biến gốc trước
    for v in range(1, graph.n + 1):
        for i in range(1, k + 1):
            _ = x_var(v, i)

    # 1. Ràng buộc ALO + AMO cho từng đỉnh v
    for v in range(1, graph.n + 1):
        lits = [x_var(v, i) for i in range(1, k + 1)]
        # ALO: Mỗi đỉnh có ít nhất 1 màu
        clauses.append(lits)
        # AMO: Mỗi đỉnh có tối đa 1 màu
        clauses.extend(build_amo_clauses(lits, amo_type, vpool))

    # 2. Ràng buộc cạnh kề: hai đỉnh kề nhau không cùng màu
    for u, v in graph.edges:
        for i in range(1, k + 1):
            clauses.append([-x_var(u, i), -x_var(v, i)])

    # 3. Phá vỡ đối xứng cơ bản trên một clique nhỏ (neo màu cho 2 đỉnh của cạnh đầu tiên)
    if graph.edges and k >= 2:
        u0, v0 = min(graph.edges)
        clauses.append([x_var(u0, 1)])
        clauses.append([x_var(v0, 2)])

    return clauses, vpool.top, vpool


def solve_direct_sat_k(
    graph: Graph,
    k: int,
    amo_type: str = 'product',
    solver_name: str = 'cadical195',
    timeout: float = 100.0
) -> Dict:
    """Giải bài toán quyết định k-Coloring với kỹ thuật AMO chỉ định và đo thống kê."""
    t0 = time.perf_counter()
    clauses, num_vars, _ = encode_direct_k_coloring(graph, k, amo_type=amo_type)
    enc_time = time.perf_counter() - t0
    num_clauses = len(clauses)

    if enc_time >= timeout:
        return {
            'vars': num_vars, 'clauses': num_clauses,
            'enc_time': enc_time, 'solve_time': 0.0, 'total_time': enc_time,
            'status': 'Timeout', 'sat': None
        }

    t1 = time.perf_counter()
    try:
        with Solver(name=solver_name, bootstrap_with=clauses) as solver:
            sat = solver.solve()
    except Exception:
        with Solver(name='glucose4', bootstrap_with=clauses) as solver:
            sat = solver.solve()
    solve_time = time.perf_counter() - t1
    total_time = enc_time + solve_time

    status = 'SAT' if sat else 'UNSAT'
    if total_time >= timeout:
        status = 'Timeout'

    return {
        'vars': num_vars,
        'clauses': num_clauses,
        'enc_time': enc_time,
        'solve_time': solve_time,
        'total_time': total_time,
        'status': status,
        'sat': sat
    }


def solve_direct_sat_optimization(
    graph: Graph,
    amo_type: str = 'product',
    solver_name: str = 'cadical195',
    timeout: float = 300.0
) -> Dict:
    """Tìm sắc số tối ưu chi(G) bằng Direct SAT Encoding (chiến lược giảm dần từ UB hoặc tăng từ LB)."""
    t_start = time.perf_counter()
    lb = greedy_clique_lb(graph)
    ub, _ = dsatur_coloring(graph)
    best_k = ub

    total_vars, total_clauses = 0, 0
    for k in range(lb, ub + 1):
        elapsed = time.perf_counter() - t_start
        if elapsed >= timeout:
            return {'chi': best_k, 'status': 'FEAS', 'time': elapsed, 'vars': total_vars, 'clauses': total_clauses}
        res = solve_direct_sat_k(graph, k, amo_type=amo_type, solver_name=solver_name, timeout=timeout - elapsed)
        total_vars = max(total_vars, res['vars'])
        total_clauses = max(total_clauses, res['clauses'])
        if res['status'] == 'Timeout':
            return {'chi': best_k, 'status': 'FEAS', 'time': time.perf_counter() - t_start, 'vars': total_vars, 'clauses': total_clauses}
        if res['sat']:
            best_k = k
            return {'chi': best_k, 'status': 'OPT', 'time': time.perf_counter() - t_start, 'vars': total_vars, 'clauses': total_clauses}

    return {'chi': best_k, 'status': 'OPT', 'time': time.perf_counter() - t_start, 'vars': total_vars, 'clauses': total_clauses}


# ==============================================================================
# PHẦN 4: MÃ HÓA CÂY ZYKOV (FULL ENCODING & CEGAR INCREMENTAL SAT)
# Cài đặt chính xác theo bài báo Glorian et al. (CP 2019)
# ==============================================================================

class ZykovSATModel:
    """
    Quản lý biến và mệnh đề cho mô hình ghép cặp đỉnh trên cây Zykov:
    - Biến s_{i,j} (với 1 <= i < j <= n): đỉnh i và j được tô cùng màu (co đỉnh).
    - Biến c_j (với 2 <= j <= n): đỉnh j cần một màu mới, tức c_j <=> AND_{i < j} (-s_{i,j}).
    """
    def __init__(self, graph: Graph, full_transitivity: bool = False):
        self.graph = graph
        self.n = graph.n
        self.vpool = IDPool()
        self.s_vars: Dict[Tuple[int, int], int] = {}
        self.c_vars: Dict[int, int] = {}
        self.base_clauses: List[List[int]] = []

        # Khởi tạo biến s_{i,j} cho 1 <= i < j <= n
        for i in range(1, self.n + 1):
            for j in range(i + 1, self.n + 1):
                self.s_vars[(i, j)] = self.vpool.id(f"s_{i}_{j}")

        # Khởi tạo biến c_j cho 2 <= j <= n
        for j in range(2, self.n + 1):
            self.c_vars[j] = self.vpool.id(f"c_{j}")

        # 1. Mệnh đề đơn cho các cạnh: -s_{u,v} forall (u,v) in E
        for u, v in graph.edges:
            self.base_clauses.append([-self.s(u, v)])

        # 2. Định nghĩa biến đếm màu mới c_j <=> AND_{i < j} (-s_{i,j}) (Equation 2 trong bài báo)
        for j in range(2, self.n + 1):
            cj = self.c_vars[j]
            # c_j => -s_{i,j}  tương đương  (-c_j v -s_{i,j})
            for i in range(1, j):
                self.base_clauses.append([-cj, -self.s(i, j)])
            # (AND_{i < j} -s_{i,j}) => c_j  tương đương  (c_j v s_{1,j} v ... v s_{j-1,j})
            self.base_clauses.append([cj] + [self.s(i, j) for i in range(1, j)])

        # 3. Nếu là Full Encoding: thêm toàn bộ mệnh đề bậc ba transitivity & euclideanity O(n^3)
        if full_transitivity:
            for i in range(1, self.n + 1):
                for j in range(i + 1, self.n + 1):
                    for k in range(j + 1, self.n + 1):
                        self.base_clauses.extend(self.get_triple_clauses(i, j, k))

    def s(self, u: int, v: int) -> int:
        if u > v:
            u, v = v, u
        return self.s_vars[(u, v)]

    def get_triple_clauses(self, u: int, v: int, w: int) -> List[List[int]]:
        """
        Trả về các mệnh đề bậc ba transitivity(i, j, k) và euclideanity(i, j, k)
        cho bộ ba đỉnh sắp thứ tự i < j < k:
        - transitivity(i,j,k): (-s_{ij} v -s_{jk} v s_{ik})
        - euclideanity(i,j,k): (-s_{ij} v -s_{ik} v s_{jk})
        (Thêm đầy đủ 3 hoán vị để đảm bảo quan hệ tương đương hoàn toàn trên bộ ba).
        """
        i, j, k = sorted((u, v, w))
        s_ij = self.s(i, j)
        s_jk = self.s(j, k)
        s_ik = self.s(i, k)
        return [
            [-s_ij, -s_jk, s_ik],  # transitivity(i, j, k)
            [-s_ij, -s_ik, s_jk],  # euclideanity(i, j, k)
            [-s_ik, -s_jk, s_ij]   # tính đối xứng bắc cầu thứ 3
        ]

    def check_and_find_spurious_triples(self, model_set: Set[int], k: int) -> Tuple[Set[Tuple[int, int, int]], Dict[int, int]]:
        """
        Cài đặt chính xác Algorithm 1: check(lambda, G=(V,E), k) trong bài báo Glorian et al. (CP 2019).
        Trả về: (Tập bộ ba T vi phạm cần tinh chỉnh, Phép tô màu hợp lệ nếu T rỗng).
        """
        T: Set[Tuple[int, int, int]] = set()
        # Dòng 2: for u <- 1 to |V| do c[u] <- {1, 2, ..., k}
        c: Dict[int, Set[int]] = {u: set(range(1, k + 1)) for u in range(1, self.n + 1)}
        assigned_color: Dict[int, int] = {}

        # Dòng 3: for u <- 1 to |V| do
        for u in range(1, self.n + 1):
            # Dòng 4: if c[u] != empty then
            if c[u]:
                # Dòng 5: c[u] = {i} s.t. i in c[u] (chọn màu nhỏ nhất khả dụng)
                chosen_i = min(c[u])
                c[u] = {chosen_i}
                assigned_color[u] = chosen_i

                # Dòng 6: for v <- u + 1 to |V| do
                for v in range(u + 1, self.n + 1):
                    s_uv = self.s(u, v)
                    # Dòng 7: if -s_{uv} in lambda then
                    if s_uv not in model_set:
                        # Dòng 8: c[v] = c[v] \ c[u]
                        c[v].discard(chosen_i)
                        # Dòng 9: if c[v] == empty and exists w < u s.t. s_{wv} in lambda and c[w] == c[u]
                        if not c[v]:
                            for w in range(1, u):
                                if self.s(w, v) in model_set and c[w] == c[u]:
                                    T.add(tuple(sorted((u, v, w))))
                                    break
                    else:
                        # Dòng 11-12: else (s_{uv} in lambda) -> c[v] = c[v] intersect c[u]
                        c[v].intersection_update(c[u])
                        # Dòng 13: if c[v] == empty then
                        if not c[v]:
                            # Dòng 14: let w < u s.t. (s_{wv} in lambda and c[w] != c[u]) or (-s_{wv} in lambda and c[w] == c[u])
                            for w in range(1, u):
                                s_wv_true = (self.s(w, v) in model_set)
                                if (s_wv_true and c[w] != c[u]) or ((not s_wv_true) and c[w] == c[u]):
                                    T.add(tuple(sorted((u, v, w))))
                                    break

        # Kiểm tra bổ sung để đảm bảo mọi bộ ba vi phạm trực tiếp đều bị bắt nếu T chưa phát hiện
        if not T and len(assigned_color) < self.n:
            for i in range(1, self.n + 1):
                for j in range(i + 1, self.n + 1):
                    s_ij = (self.s(i, j) in model_set)
                    for w in range(j + 1, self.n + 1):
                        s_jw = (self.s(j, w) in model_set)
                        s_iw = (self.s(i, w) in model_set)
                        if (s_ij and s_jw and not s_iw) or (s_ij and s_iw and not s_jw) or (s_iw and s_jw and not s_ij):
                            T.add((i, j, w))
                            if len(T) >= 25:
                                return T, assigned_color

        return T, assigned_color


def solve_zykov_k_decision(
    zykov_model: ZykovSATModel,
    k: int,
    use_cegar: bool = True,
    solver_name: str = 'glucose4',
    timeout: float = 100.0
) -> Dict:
    """
    Giải bài toán quyết định chi(G) <= k bằng mô hình Zykov (Full hoặc CEGAR Incremental).
    Ràng buộc số màu: sum_{j=2}^n c_j <= k - 1.
    """
    t0 = time.perf_counter()
    c_lits = [zykov_model.c_vars[j] for j in range(2, zykov_model.n + 1)]

    # Ràng buộc sum_{j=2}^n c_j <= k - 1
    if k - 1 <= 0:
        card_clauses = [[-lit] for lit in c_lits]
    elif k - 1 >= len(c_lits):
        card_clauses = []
    else:
        card_cnf = CardEnc.atmost(lits=c_lits, bound=k - 1, encoding=EncType.totalizer, vpool=zykov_model.vpool)
        card_clauses = card_cnf.clauses

    all_init_clauses = zykov_model.base_clauses + card_clauses
    enc_time = time.perf_counter() - t0
    solve_time = 0.0
    check_time = 0.0
    cegar_loops = 0
    added_triples: Set[Tuple[int, int, int]] = set()

    try:
        solver = Solver(name=solver_name, bootstrap_with=all_init_clauses)
    except Exception:
        solver = Solver(name='cadical195', bootstrap_with=all_init_clauses)

    try:
        while True:
            if (time.perf_counter() - t0) >= timeout:
                return {
                    'sat': None, 'status': 'Timeout',
                    'vars': zykov_model.vpool.top, 'clauses': len(all_init_clauses) + len(added_triples) * 3,
                    'enc_time': enc_time, 'solve_time': solve_time, 'check_time': check_time,
                    'total_time': time.perf_counter() - t0, 'cegar_loops': cegar_loops
                }

            cegar_loops += 1
            ts = time.perf_counter()
            is_sat = solver.solve()
            solve_time += (time.perf_counter() - ts)

            if not is_sat:
                return {
                    'sat': False, 'status': 'UNSAT',
                    'vars': zykov_model.vpool.top, 'clauses': len(all_init_clauses) + len(added_triples) * 3,
                    'enc_time': enc_time, 'solve_time': solve_time, 'check_time': check_time,
                    'total_time': time.perf_counter() - t0, 'cegar_loops': cegar_loops
                }

            if not use_cegar:
                # Full Encoding: nghiệm SAT trả về là hợp lệ ngay lập tức
                return {
                    'sat': True, 'status': 'SAT',
                    'vars': zykov_model.vpool.top, 'clauses': len(all_init_clauses),
                    'enc_time': enc_time, 'solve_time': solve_time, 'check_time': 0.0,
                    'total_time': time.perf_counter() - t0, 'cegar_loops': 1
                }

            # CEGAR Mode: Kiểm tra phản ví dụ bằng Algorithm 1
            tc = time.perf_counter()
            model = solver.get_model()
            model_set = {lit for lit in model if lit > 0}
            spurious_triples, coloring = zykov_model.check_and_find_spurious_triples(model_set, k)
            check_time += (time.perf_counter() - tc)

            if not spurious_triples:
                # Đã tìm thấy phép tô màu k màu hợp lệ!
                actual_used = len(set(coloring.values())) if coloring else k
                return {
                    'sat': True, 'status': 'SAT', 'actual_k': min(k, actual_k if (actual_k := actual_used) > 0 else k),
                    'vars': zykov_model.vpool.top, 'clauses': len(all_init_clauses) + len(added_triples) * 3,
                    'enc_time': enc_time, 'solve_time': solve_time, 'check_time': check_time,
                    'total_time': time.perf_counter() - t0, 'cegar_loops': cegar_loops
                }

            # Tinh chỉnh (Refinement): Thêm các mệnh đề transitivity & euclideanity cho các bộ ba vi phạm
            for triple in spurious_triples:
                if triple not in added_triples:
                    added_triples.add(triple)
                    for cl in zykov_model.get_triple_clauses(*triple):
                        solver.add_clause(cl)
                        zykov_model.base_clauses.append(cl)
    finally:
        solver.delete()


def solve_zykov_optimization(
    graph: Graph,
    use_cegar: bool = True,
    search_strategy: str = '1toN',
    solver_name: str = 'glucose4',
    timeout: float = 300.0
) -> Dict:
    """
    Tìm sắc số tối ưu chi(G) bằng mô hình Zykov (Full hoặc CEGAR Incremental)
    với 3 chiến lược tìm kiếm cận trong bài báo:
    - '1toN' : Tăng dần từ cận dưới LB lên UB (Ascending search)
    - 'Nto1' : Giảm dần từ cận trên UB xuống LB (Descending search)
    - 'Dicho': Tìm kiếm nhị phân (Dichotomic / Binary search)
    """
    t_start = time.perf_counter()
    lb = greedy_clique_lb(graph)
    ub, _ = dsatur_coloring(graph)

    zykov_model = ZykovSATModel(graph, full_transitivity=(not use_cegar))
    init_vars = zykov_model.vpool.top
    init_clauses = len(zykov_model.base_clauses)

    total_enc = time.perf_counter() - t_start
    total_solve = 0.0
    total_check = 0.0
    total_loops = 0
    best_chi = ub
    status = 'OPT'

    if lb == ub:
        return {
            'chi': best_chi, 'status': 'OPT',
            'vars': init_vars, 'clauses': init_clauses,
            'enc_time': total_enc, 'solve_time': 0.0, 'check_time': 0.0,
            'total_time': time.perf_counter() - t_start, 'cegar_loops': 0
        }

    if search_strategy == '1toN':
        for k in range(lb, ub):
            elapsed = time.perf_counter() - t_start
            if elapsed >= timeout:
                status = 'FEAS'
                break
            res = solve_zykov_k_decision(zykov_model, k, use_cegar=use_cegar, solver_name=solver_name, timeout=timeout - elapsed)
            total_enc += res['enc_time']
            total_solve += res['solve_time']
            total_check += res['check_time']
            total_loops += res['cegar_loops']
            if res['status'] == 'Timeout':
                status = 'FEAS'
                break
            if res['sat']:
                best_chi = k
                status = 'OPT'
                break

    elif search_strategy == 'Nto1':
        k = ub - 1
        while k >= lb:
            elapsed = time.perf_counter() - t_start
            if elapsed >= timeout:
                status = 'FEAS'
                break
            res = solve_zykov_k_decision(zykov_model, k, use_cegar=use_cegar, solver_name=solver_name, timeout=timeout - elapsed)
            total_enc += res['enc_time']
            total_solve += res['solve_time']
            total_check += res['check_time']
            total_loops += res['cegar_loops']
            if res['status'] == 'Timeout':
                status = 'FEAS'
                break
            if res['sat']:
                best_chi = res.get('actual_k', k)
                k = best_chi - 1
            else:
                status = 'OPT'
                break

    elif search_strategy == 'Dicho':
        low, high = lb, ub
        while low < high:
            elapsed = time.perf_counter() - t_start
            if elapsed >= timeout:
                status = 'FEAS'
                break
            mid = (low + high) // 2
            res = solve_zykov_k_decision(zykov_model, mid, use_cegar=use_cegar, solver_name=solver_name, timeout=timeout - elapsed)
            total_enc += res['enc_time']
            total_solve += res['solve_time']
            total_check += res['check_time']
            total_loops += res['cegar_loops']
            if res['status'] == 'Timeout':
                status = 'FEAS'
                break
            if res['sat']:
                best_chi = res.get('actual_k', mid)
                high = best_chi
            else:
                low = mid + 1

    return {
        'chi': best_chi,
        'status': status,
        'vars': zykov_model.vpool.top,
        'clauses': len(zykov_model.base_clauses),
        'enc_time': total_enc,
        'solve_time': total_solve,
        'check_time': total_check,
        'total_time': time.perf_counter() - t_start,
        'cegar_loops': total_loops
    }


# ==============================================================================
# PHẦN 5: CÁC BỘ GIẢI CÔNG NGHIỆP (OR-TOOLS CP-SAT, GUROBI ILP, CPLEX MIP/CP)
# ==============================================================================

def solve_ortools_cpsat(graph: Graph, timeout: float = 300.0) -> Dict:
    """Giải bài toán GCP bằng Google OR-Tools CP-SAT."""
    if not HAS_ORTOOLS:
        return {'chi': None, 'status': 'NotInstalled', 'time': None}

    t0 = time.perf_counter()
    ub, _ = dsatur_coloring(graph)
    lb = greedy_clique_lb(graph)

    model = cp_model.CpModel()
    q = {v: model.NewIntVar(1, ub, f"q_{v}") for v in range(1, graph.n + 1)}
    max_color = model.NewIntVar(lb, ub, "max_color")

    for u, v in graph.edges:
        model.Add(q[u] != q[v])

    for v in range(1, graph.n + 1):
        model.Add(max_color >= q[v])

    # Neo đối xứng cho 1 cạnh đầu tiên
    if graph.edges:
        u0, v0 = min(graph.edges)
        model.Add(q[u0] == 1)
        model.Add(q[v0] == 2)

    model.Minimize(max_color)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1.0, timeout)
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)
    total_time = time.perf_counter() - t0

    if status == cp_model.OPTIMAL:
        return {'chi': int(solver.ObjectiveValue()), 'status': 'OPT', 'time': total_time}
    elif status == cp_model.FEASIBLE:
        return {'chi': int(solver.ObjectiveValue()), 'status': 'FEAS', 'time': total_time}
    else:
        return {'chi': ub, 'status': 'Timeout', 'time': total_time}


def solve_gurobi_ilp(graph: Graph, timeout: float = 300.0) -> Dict:
    """Giải bài toán GCP bằng mô hình ILP trên Gurobi Optimizer."""
    if not HAS_GUROBI:
        return {'chi': None, 'status': 'NotInstalled', 'time': None}

    t0 = time.perf_counter()
    ub, _ = dsatur_coloring(graph)
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("GCP_ILP", env=env)
        model.setParam(GRB.Param.TimeLimit, timeout)

        w = model.addVars(range(1, ub + 1), vtype=GRB.BINARY, name="w")
        x = model.addVars(range(1, graph.n + 1), range(1, ub + 1), vtype=GRB.BINARY, name="x")

        # Mỗi đỉnh nhận đúng 1 màu
        for v in range(1, graph.n + 1):
            model.addConstr(gp.quicksum(x[v, i] for i in range(1, ub + 1)) == 1)

        # Hai đỉnh kề nhau không cùng màu & kích hoạt biến màu w_i
        for u, v in graph.edges:
            for i in range(1, ub + 1):
                model.addConstr(x[u, i] + x[v, i] <= w[i])

        # Phá vỡ đối xứng thứ tự màu: w_i >= w_{i+1}
        for i in range(1, ub):
            model.addConstr(w[i] >= w[i + 1])

        model.setObjective(gp.quicksum(w[i] for i in range(1, ub + 1)), GRB.MINIMIZE)
        model.optimize()
        total_time = time.perf_counter() - t0

        if model.Status == GRB.OPTIMAL:
            return {'chi': int(round(model.ObjVal)), 'status': 'OPT', 'time': total_time}
        elif model.SolCount > 0:
            return {'chi': int(round(model.ObjVal)), 'status': 'FEAS', 'time': total_time}
        else:
            return {'chi': ub, 'status': 'Timeout', 'time': total_time}
    except Exception as e:
        return {'chi': None, 'status': f'Error ({type(e).__name__})', 'time': time.perf_counter() - t0}


def solve_cplex_mip(graph: Graph, timeout: float = 300.0) -> Dict:
    """Giải bài toán GCP bằng IBM ILOG CPLEX MIP."""
    if not HAS_CPLEX:
        return {'chi': None, 'status': 'NotInstalled', 'time': None}

    t0 = time.perf_counter()
    ub, _ = dsatur_coloring(graph)
    try:
        model = CplexMIPModel(name="GCP_CPLEX_MIP")
        model.set_time_limit(timeout)
        w = model.binary_var_dict(range(1, ub + 1), name="w")
        x = model.binary_var_matrix(range(1, graph.n + 1), range(1, ub + 1), name="x")

        for v in range(1, graph.n + 1):
            model.add_constraint(model.sum(x[v, i] for i in range(1, ub + 1)) == 1)
        for u, v in graph.edges:
            for i in range(1, ub + 1):
                model.add_constraint(x[u, i] + x[v, i] <= w[i])
        for i in range(1, ub):
            model.add_constraint(w[i] >= w[i + 1])

        model.minimize(model.sum(w[i] for i in range(1, ub + 1)))
        sol = model.solve(log_output=False)
        total_time = time.perf_counter() - t0
        if sol:
            st = 'OPT' if 'optimal' in str(model.solve_details.status).lower() else 'FEAS'
            return {'chi': int(round(sol.objective_value)), 'status': st, 'time': total_time}
        return {'chi': ub, 'status': 'Timeout', 'time': total_time}
    except Exception:
        return {'chi': None, 'status': 'Error', 'time': time.perf_counter() - t0}


def solve_cplex_cp(graph: Graph, timeout: float = 300.0) -> Dict:
    """Giải bài toán GCP bằng IBM ILOG CPLEX CP Optimizer."""
    if not HAS_CPLEX:
        return {'chi': None, 'status': 'NotInstalled', 'time': None}

    t0 = time.perf_counter()
    ub, _ = dsatur_coloring(graph)
    try:
        model = CplexCPModel()
        q = {v: model.integer_var(1, ub, name=f"q_{v}") for v in range(1, graph.n + 1)}
        for u, v in graph.edges:
            model.add(q[u] != q[v])
        model.add(model.minimize(model.max([q[v] for v in range(1, graph.n + 1)])))
        sol = model.solve(TimeLimit=timeout, LogVerbosity='Quiet')
        total_time = time.perf_counter() - t0
        if sol:
            st = 'OPT' if sol.is_solution_optimal() else 'FEAS'
            return {'chi': int(round(sol.get_objective_values()[0])), 'status': st, 'time': total_time}
        return {'chi': ub, 'status': 'Timeout', 'time': total_time}
    except Exception:
        return {'chi': None, 'status': 'Error', 'time': time.perf_counter() - t0}


# ==============================================================================
# PHẦN 6: PIPELINE CHẠY THỰC NGHIỆM, XUẤT BẢNG LATEX & VẼ BIỂU ĐỒ
# ==============================================================================

def run_full_benchmark_pipeline(
    graphs: List[Graph],
    amo_timeout: float = 60.0,
    solver_timeout: float = 120.0,
    output_dir: str = "results"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chạy toàn bộ thực nghiệm cho Bảng 1, Bảng 2 và Bảng 4 trong báo cáo,
    lưu ra CSV, in mã nguồn bảng LaTeX và vẽ các biểu đồ.
    """
    os.makedirs(output_dir, exist_ok=True)
    amo_methods = ['pairwise', 'seqcounter', 'binary', 'commander', 'product']

    amo_records = []
    solver_records = []

    print("=" * 85)
    print("BẮT ĐẦU CHẠY THỰC NGHIỆM BÀI TOÁN TÔ MÀU ĐỒ THỊ (GCP)")
    print("=" * 85)

    for g in graphs:
        ub, _ = dsatur_coloring(g)
        lb = greedy_clique_lb(g)
        target_k = g.known_chi if g.known_chi is not None else ub
        print(f"\n>>> Đồ thị: {g.name} (|V|={g.n}, |E|={g.m}, LB={lb}, DSatur UB={ub}, Target k={target_k})")

        # 1. Đánh giá 5 kỹ thuật mã hóa AMO ở ngưỡng quyết định k = target_k
        row_amo = {
            'Instance': g.name,
            '|V|': g.n,
            '|E|': g.m,
            'k': target_k
        }
        for amo in amo_methods:
            res_amo = solve_direct_sat_k(g, target_k, amo_type=amo, timeout=amo_timeout)
            row_amo[f'{amo}_vars'] = res_amo['vars']
            row_amo[f'{amo}_clauses'] = res_amo['clauses']
            row_amo[f'{amo}_time'] = res_amo['total_time']
            row_amo[f'{amo}_status'] = res_amo['status']
            print(f"  [Direct AMO: {amo:10s}] Vars={res_amo['vars']:7d} | Clauses={res_amo['clauses']:9d} | Time={res_amo['total_time']:.4f}s ({res_amo['status']})")

        # 2. Chạy Zykov Full (1toN) và Zykov CEGAR Incremental (1toN)
        res_zykov_full = solve_zykov_optimization(g, use_cegar=False, search_strategy='1toN', timeout=amo_timeout)
        print(f"  [Zykov Full (1toN)     ] Vars={res_zykov_full['vars']:7d} | Clauses={res_zykov_full['clauses']:9d} | Time={res_zykov_full['total_time']:.4f}s | chi={res_zykov_full['chi']} ({res_zykov_full['status']})")

        res_zykov_cegar = solve_zykov_optimization(g, use_cegar=True, search_strategy='1toN', timeout=solver_timeout)
        print(f"  [Zykov CEGAR (1toN)    ] Vars={res_zykov_cegar['vars']:7d} | Clauses={res_zykov_cegar['clauses']:9d} | Time={res_zykov_cegar['total_time']:.4f}s | Loops={res_zykov_cegar['cegar_loops']} | chi={res_zykov_cegar['chi']} ({res_zykov_cegar['status']})")

        row_amo['zykov_full_vars'] = res_zykov_full['vars']
        row_amo['zykov_full_clauses'] = res_zykov_full['clauses']
        row_amo['zykov_full_time'] = res_zykov_full['total_time']
        row_amo['zykov_full_status'] = res_zykov_full['status']

        row_amo['zykov_cegar_vars'] = res_zykov_cegar['vars']
        row_amo['zykov_cegar_clauses'] = res_zykov_cegar['clauses']
        row_amo['zykov_cegar_time'] = res_zykov_cegar['total_time']
        row_amo['zykov_cegar_status'] = res_zykov_cegar['status']
        amo_records.append(row_amo)

        # 3. So sánh với Direct SAT (Product), OR-Tools CP-SAT, Gurobi, CPLEX
        res_direct_opt = solve_direct_sat_optimization(g, amo_type='product', timeout=solver_timeout)
        res_ortools = solve_ortools_cpsat(g, timeout=solver_timeout)
        res_gurobi = solve_gurobi_ilp(g, timeout=solver_timeout)
        res_cplex_mip = solve_cplex_mip(g, timeout=solver_timeout)
        res_cplex_cp = solve_cplex_cp(g, timeout=solver_timeout)

        ort_t = res_ortools['time'] if res_ortools['time'] is not None else -1.0
        print(f"  [OR-Tools CP-SAT       ] Time={ort_t:.4f}s | chi={res_ortools['chi']} ({res_ortools['status']})")

        solver_records.append({
            'Instance': g.name,
            '|V|': g.n,
            '|E|': g.m,
            'chi': res_zykov_cegar['chi'],
            'Zykov_CEGAR_time': res_zykov_cegar['total_time'],
            'Zykov_CEGAR_status': res_zykov_cegar['status'],
            'PySAT_Product_time': res_direct_opt['time'],
            'PySAT_Product_status': res_direct_opt['status'],
            'OR_Tools_CPSAT_time': res_ortools['time'],
            'OR_Tools_CPSAT_status': res_ortools['status'],
            'Gurobi_ILP_time': res_gurobi['time'],
            'Gurobi_ILP_status': res_gurobi['status'],
            'CPLEX_MIP_time': res_cplex_mip['time'],
            'CPLEX_MIP_status': res_cplex_mip['status'],
            'CPLEX_CP_time': res_cplex_cp['time'],
            'CPLEX_CP_status': res_cplex_cp['status'],
        })

    df_amo = pd.DataFrame(amo_records)
    df_solvers = pd.DataFrame(solver_records)

    amo_csv = os.path.join(output_dir, "amo_encoding_results.csv")
    solvers_csv = os.path.join(output_dir, "solver_benchmark_results.csv")
    df_amo.to_csv(amo_csv, index=False)
    df_solvers.to_csv(solvers_csv, index=False)
    print(f"\n[SAVED] Đã lưu kết quả CSV vào '{amo_csv}' và '{solvers_csv}'.")

    # Sinh file chứa các bảng LaTeX để copy vào báo cáo
    export_latex_tables(df_amo, df_solvers, os.path.join(output_dir, "latex_tables_generated.tex"))

    # Vẽ 3 biểu đồ cho báo cáo
    plot_report_figures(df_amo, df_solvers, output_dir=output_dir)

    return df_amo, df_solvers


def export_latex_tables(df_amo: pd.DataFrame, df_solvers: pd.DataFrame, filepath: str) -> None:
    """Xuất các hàng dữ liệu định dạng LaTeX chuẩn xác để dán thẳng vào Bảng 1, 2, 4 của báo cáo."""
    bs = chr(92)  # Ký tự backslash '\'
    nl = chr(10)  # Ký tự xuống dòng '\n'
    lines = []
    lines.append("% ==================== BẢNG 1: SỐ BIẾN / SỐ MỆNH ĐỀ CNF ====================")
    for _, r in df_amo.iterrows():
        inst = str(r['Instance']).replace('_', bs + '_')
        row_str = (
            f"{bs}texttt{{{inst}}} (${r['|V|']}, {r['|E|']}, {r['k']}$) & "
            f"{r['pairwise_vars']:,} / {r['pairwise_clauses']:,} & "
            f"{r['seqcounter_vars']:,} / {r['seqcounter_clauses']:,} & "
            f"{r['binary_vars']:,} / {r['binary_clauses']:,} & "
            f"{r['commander_vars']:,} / {r['commander_clauses']:,} & "
            f"{r['product_vars']:,} / {r['product_clauses']:,} & "
            f"{r['zykov_cegar_vars']:,} / {r['zykov_cegar_clauses']:,} {bs}{bs}"
        )
        lines.append(row_str)

    lines.append("")
    lines.append("% ==================== BẢNG 2: THỜI GIAN CHẠY CÁC KỸ THUẬT SAT ====================")
    for _, r in df_amo.iterrows():
        inst = str(r['Instance']).replace('_', bs + '_')
        row_str = (
            f"{bs}texttt{{{inst}}} & "
            f"{r['pairwise_time']:.4f} & "
            f"{r['seqcounter_time']:.4f} & "
            f"{r['binary_time']:.4f} & "
            f"{r['commander_time']:.4f} & "
            f"{r['product_time']:.4f} & "
            f"{r['zykov_full_time']:.4f} & "
            f"{bs}textbf{{{r['zykov_cegar_time']:.4f}}} {bs}{bs}"
        )
        lines.append(row_str)

    lines.append("")
    lines.append("% ==================== BẢNG 4: SO SÁNH SAT VS ILP VS CP ====================")
    for _, r in df_solvers.iterrows():
        inst = str(r['Instance']).replace('_', bs + '_')
        ort_str = f"{r['OR_Tools_CPSAT_time']:.4f}" if pd.notna(r['OR_Tools_CPSAT_time']) else r['OR_Tools_CPSAT_status']
        grb_str = f"{r['Gurobi_ILP_time']:.4f}" if pd.notna(r['Gurobi_ILP_time']) else r['Gurobi_ILP_status']
        mip_str = f"{r['CPLEX_MIP_time']:.4f}" if pd.notna(r['CPLEX_MIP_time']) else r['CPLEX_MIP_status']
        cp_str = f"{r['CPLEX_CP_time']:.4f}" if pd.notna(r['CPLEX_CP_time']) else r['CPLEX_CP_status']
        row_str = (
            f"{bs}texttt{{{inst}}} (${bs}chi = {r['chi']}$) & "
            f"{bs}textbf{{{r['Zykov_CEGAR_time']:.4f}}} & "
            f"{r['PySAT_Product_time']:.4f} & "
            f"{grb_str} & {mip_str} & {cp_str} & {ort_str} {bs}{bs}"
        )
        lines.append(row_str)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(nl.join(lines) + nl)
    print(f"[SAVED] Đã xuất mã nguồn các bảng LaTeX tại '{filepath}'.")


def plot_report_figures(df_amo: pd.DataFrame, df_solvers: pd.DataFrame, output_dir: str = "results") -> None:
    """
    Sinh 3 file hình ảnh chuẩn cho báo cáo LaTeX:
    1. fig_workflow.png        : Sơ đồ luồng kiến trúc giải Bài toán GCP bằng SAT Solving.
    2. fig_clause_growth.png   : Tốc độ tăng trưởng mệnh đề CNF (Log Scale) giữa các kỹ thuật mã hóa.
    3. fig_solver_benchmark.png: Đường cong thời gian thực thi (Log Scale) giữa SAT, CP và ILP.
    """
    nl = chr(10)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    # --------------------------------------------------------------------------
    # HÌNH 1: SƠ ĐỒ LUỒNG KIẾN TRÚC (fig_workflow.png)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 2.8), dpi=300)
    ax.axis('off')
    boxes = [
        ("Đồ thị đầu vào G=(V,E)" + nl + "& Khởi tạo cận (DSatur/Clique)", "#E3F2FD", "#1E88E5"),
        ("Mã hóa SAT CNF" + nl + "(Direct AMO / Cây Zykov)", "#FFF3E0", "#FB8C00"),
        ("Giải lặp Incremental SAT" + nl + "(CDCL Solver + CEGAR Check)", "#E8F5E9", "#43A047"),
        ("Xuất Sắc số Tối ưu χ(G)" + nl + "& Phân hoạch màu các đỉnh", "#F3E5F5", "#8E24AA")
    ]
    x_positions = [0.12, 0.37, 0.63, 0.88]
    for idx, (text, facecolor, edgecolor) in enumerate(boxes):
        x = x_positions[idx]
        bbox_props = dict(boxstyle="round,pad=0.65,rounding_size=0.2", fc=facecolor, ec=edgecolor, lw=2.0)
        ax.text(x, 0.5, text, ha="center", va="center", fontsize=10, fontweight="bold", color="#1A1A1A", bbox=bbox_props)
        if idx < len(boxes) - 1:
            ax.annotate(
                "", xy=(x_positions[idx + 1] - 0.105, 0.5), xytext=(x + 0.105, 0.5),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", lw=2.0, color="#424242")
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    workflow_path = os.path.join(output_dir, "fig_workflow.png")
    plt.savefig(workflow_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --------------------------------------------------------------------------
    # HÌNH 2: TỐC ĐỘ TĂNG TRƯỞNG MỆNH ĐỀ CNF (fig_clause_growth.png)
    # --------------------------------------------------------------------------
    df_plot = df_amo.sort_values(by='|V|').reset_index(drop=True)
    x_labels = [f"{r['Instance']}{nl}(|V|={r['|V|']})" for _, r in df_plot.iterrows()]
    x_idx = np.arange(len(df_plot))

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    ax.plot(x_idx, df_plot['zykov_full_clauses'], 's--', color='#8E24AA', lw=2.0, label='Zykov Full Encoding O(|V|³)')
    ax.plot(x_idx, df_plot['pairwise_clauses'], 'o-', color='#E53935', lw=2.0, label='Direct Pairwise / Binomial')
    ax.plot(x_idx, df_plot['binary_clauses'], '^-', color='#FB8C00', lw=1.8, label='Direct Binary Encoding')
    ax.plot(x_idx, df_plot['seqcounter_clauses'], 'v-', color='#00ACC1', lw=1.8, label='Direct Sequential Counter')
    ax.plot(x_idx, df_plot['commander_clauses'], 'd-', color='#43A047', lw=1.8, label='Direct Commander Encoding')
    ax.plot(x_idx, df_plot['product_clauses'], 'P-', color='#1E88E5', lw=2.0, label='Direct Product Encoding')
    ax.plot(x_idx, df_plot['zykov_cegar_clauses'], '*-', color='#D81B60', lw=2.4, markersize=9, label='Zykov CEGAR Incremental (Tối ưu)')

    ax.set_yscale('log')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel('Bộ dữ liệu đồ thị thực nghiệm (Sắp xếp theo số đỉnh |V|)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Số lượng Mệnh đề CNF (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_title('So sánh Số lượng Mệnh đề CNF giữa các Kỹ thuật Mã hóa SAT', fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.45)
    ax.legend(loc='best', fontsize=8.8, frameon=True)
    plt.tight_layout()
    clause_fig_path = os.path.join(output_dir, "fig_clause_growth.png")
    plt.savefig(clause_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --------------------------------------------------------------------------
    # HÌNH 3: SO SÁNH THỜI GIAN GIẢI THỰC TẾ (fig_solver_benchmark.png)
    # --------------------------------------------------------------------------
    df_s = df_solvers.sort_values(by='|V|').reset_index(drop=True)
    x_s_labels = [f"{r['Instance']}{nl}(χ={r['chi']})" for _, r in df_s.iterrows()]
    x_s_idx = np.arange(len(df_s))

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    ax.plot(x_s_idx, df_s['Zykov_CEGAR_time'], 'o-', color='#1E88E5', lw=2.4, markersize=7, label='PySAT (Zykov CEGAR 1toN)')
    ax.plot(x_s_idx, df_s['PySAT_Product_time'], 's--', color='#00897B', lw=2.0, markersize=6, label='PySAT Direct (Product AMO)')

    if df_s['OR_Tools_CPSAT_time'].notna().any():
        ax.plot(x_s_idx, df_s['OR_Tools_CPSAT_time'], '^-', color='#43A047', lw=2.0, markersize=6, label='Google OR-Tools (CP-SAT)')
    if df_s['Gurobi_ILP_time'].notna().any():
        ax.plot(x_s_idx, df_s['Gurobi_ILP_time'], 'x-', color='#E53935', lw=2.0, markersize=7, label='Gurobi Optimizer (ILP)')

    ax.set_yscale('log')
    ax.set_xticks(x_s_idx)
    ax.set_xticklabels(x_s_labels, fontsize=9)
    ax.set_xlabel('Bộ dữ liệu đồ thị thực nghiệm (Kèm sắc số tối ưu χ(G))', fontsize=11, fontweight='bold')
    ax.set_ylabel('Thời gian thực thi - Giây (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_title('So sánh Thời gian Giải thực tế giữa các Bộ giải (SAT vs CP vs ILP)', fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.45)
    ax.legend(loc='best', fontsize=9.5, frameon=True)
    plt.tight_layout()
    solver_fig_path = os.path.join(output_dir, "fig_solver_benchmark.png")
    plt.savefig(solver_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVED] Đã xuất 3 biểu đồ PNG vào thư mục '{output_dir}/':")
    print(f"  - {workflow_path}")
    print(f"  - {clause_fig_path}")
    print(f"  - {solver_fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Runner cho Bài toán Tô màu Đồ thị (GCP)")
    parser.add_argument("--data-dir", type=str, default="data", help="Thư mục chứa các file đồ thị .col")
    parser.add_argument("--output-dir", type=str, default="results", help="Thư mục xuất CSV, bảng LaTeX và biểu đồ")
    parser.add_argument("--amo-timeout", type=float, default=60.0, help="Timeout (giây) cho đánh giá AMO")
    parser.add_argument("--solver-timeout", type=float, default=120.0, help="Timeout (giây) cho so sánh các bộ giải")
    args = parser.parse_args()

    benchmark_graphs = load_benchmark_suite(args.data_dir)
    run_full_benchmark_pipeline(
        benchmark_graphs,
        amo_timeout=args.amo_timeout,
        solver_timeout=args.solver_timeout,
        output_dir=args.output_dir
    )