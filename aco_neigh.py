# aco_neighborhood.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import math, random, time
import networkx as nx

# -------------------------------
# Modelo de grafo y demandas
# -------------------------------
@dataclass
class Demand:
    s: str
    d: str
    bw: int            # slots requeridos
    cls: str           # 'gold' | 'silver' | 'bronze'
    delay_max: float
    hops_max: int

# Cada arista e tiene atributos:
#  'delay': float (ms), 'dist': float (km),
#  'slots': List[int] (0/1 por sub-slot),
#  'osnr': float (dB), 'srlg': int
# capas de feromonas:
#  'tau_r': float (local), 'tau_b': float (borde/gateway)

# -------------------------------
# Heurística óptica η(i,j)
# -------------------------------
def eta(g: nx.Graph, u: str, v: str, bw: int, theta_osnr: float, srlg_primary: Optional[int]) -> float:
    delay = g[u][v]['delay']
    osnr  = g[u][v]['osnr']
    slots_arr = g[u][v]['slots']
    contig = max_contiguous_ones(slots_arr)
    # penalización SRLG (si comparte riesgo con ruta primaria)
    lam_srlg = 0.5 if (srlg_primary is not None and g[u][v]['srlg'] == srlg_primary) else 1.0
    # clip sigmoidal para OSNR
    osnr_fac = 1 / (1 + math.exp(-(osnr - theta_osnr)))  # ~0 si bajo umbral, ~1 si alto
    return (1.0 / max(1e-9, delay)) * osnr_fac * (1.0 + contig / max(1, len(slots_arr))) * lam_srlg

def max_contiguous_ones(bits: List[int]) -> int:
    m = c = 0
    for b in bits:
        if b == 1:
            c += 1; m = max(m, c)
        else:
            c = 0
    return m

# -------------------------------
# Vecindario k-hop y “gateways”
# -------------------------------
def neighborhood_khop(G: nx.Graph, s: str, d: str, failed_edge: Tuple[str,str], k: int) -> nx.Graph:
    """Construye subgrafo alrededor de s y d, union de bolas k-hop,
    excluyendo el enlace fallado."""
    Gp = G.copy()
    if Gp.has_edge(*failed_edge):
        Gp.remove_edge(*failed_edge)
    nodes_s = set(nx.single_source_shortest_path_length(Gp, s, cutoff=k).keys())
    nodes_d = set(nx.single_source_shortest_path_length(Gp, d, cutoff=k).keys())
    nodes = nodes_s.union(nodes_d)
    # Garantiza conectividad mínima: si queda vacío, devuelve G' completo (último recurso)
    if not nodes: 
        return Gp
    return Gp.subgraph(nodes).copy()

def is_gateway(G: nx.Graph, H: nx.Graph, u: str) -> bool:
    """Nodo de H que toca aristas hacia G\H (borde)."""
    return any(n not in H for n in G.neighbors(u))

# -------------------------------
# Listas de candidatos por nodo
# -------------------------------
def build_candidate_lists(H: nx.Graph, L: int, dmd: Demand, theta_osnr: float, srlg_primary: Optional[int]) -> Dict[str, List[str]]:
    cand: Dict[str, List[str]] = {}
    for u in H.nodes():
        neighs = []
        for v in H.neighbors(u):
            score = eta(H, u, v, dmd.bw, theta_osnr, srlg_primary)
            neighs.append((score, v))
        neighs.sort(key=lambda x: x[0], reverse=True)
        cand[u] = [v for _, v in neighs[:L]]
    return cand

# -------------------------------
# Verificación RS/RSA/OSNR
# -------------------------------
def feasible_RS_RSA_OSNR(H: nx.Graph, path: List[str], dmd: Demand, theta_osnr: float) -> bool:
    # Delay y hops
    total_delay = sum(H[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
    if total_delay > dmd.delay_max or (len(path)-1) > dmd.hops_max:
        return False
    # Continuidad/contigüidad: intersección de bitmaps
    slots = None
    for i in range(len(path)-1):
        arr = H[path[i]][path[i+1]]['slots']
        slots = arr if slots is None else [a & b for a,b in zip(slots, arr)]
    if slots is None or max_contiguous_ones(slots) < dmd.bw:
        return False
    # OSNR mínima (modelo simple: min por salto)
    min_osnr = min(H[path[i]][path[i+1]]['osnr'] for i in range(len(path)-1))
    if min_osnr < theta_osnr:
        return False
    return True

# -------------------------------
# Construcción de P’ (ACS con vecindario)
# -------------------------------
def construct_path_with_candidates(H: nx.Graph, dmd: Demand, cand: Dict[str, List[str]],
                                  alpha: float, beta: float, tau_r: Dict[Tuple[str,str], float],
                                  tau_b: Dict[Tuple[str,str], float]) -> Optional[List[str]]:
    s, d = dmd.s, dmd.d
    current = s
    visited: Set[str] = {s}
    path = [s]
    MAX_STEPS = len(H) + 5
    for _ in range(MAX_STEPS):
        if current == d: 
            return path
        options = []
        for v in cand.get(current, []):
            if v in visited: 
                continue
            # τ = τ_r * τ_b (si es borde)
            edge = (current, v) if H.has_edge(current, v) else (v, current)
            tau_local = tau_r.get(edge, 1.0)
            tau_border = tau_b.get(edge, 1.0)
            tau = tau_local * tau_border
            nij = eta(H, current, v, dmd.bw, theta_osnr=18.0, srlg_primary=None)  # θ ejemplo
            score = (tau ** alpha) * (nij ** beta)
            options.append((score, v))
        if not options:
            return None
        # ruleta
        total = sum(s for s,_ in options)
        r = random.random() * total
        acc = 0.0
        nxt = options[0][1]
        for s_val, v in options:
            acc += s_val
            if acc >= r:
                nxt = v; break
        path.append(nxt); visited.add(nxt); current = nxt
    return None

# -------------------------------
# ACO por vecindarios (bucle principal)
# -------------------------------
def ACO_neighborhood(G: nx.Graph, dmd: Demand, failed_edge: Tuple[str,str],
                     alpha=1.0, beta=2.0, rho=0.2, L=8, k_init=2, k_max=5,
                     it_max=10, time_budget_ms=20) -> Optional[List[str]]:
    start_t = time.time()
    k = k_init
    best_path: Optional[List[str]] = None
    best_cost = float('inf')

    # feromonas
    tau_r: Dict[Tuple[str,str], float] = {}
    tau_b: Dict[Tuple[str,str], float] = {}

    while (time.time() - start_t)*1000 < time_budget_ms and k <= k_max:
        H = neighborhood_khop(G, dmd.s, dmd.d, failed_edge, k)
        # init τ si no existen
        for (u,v) in H.edges():
            tau_r.setdefault((u,v), 1.0)
            if is_gateway(G, H, u) or is_gateway(G, H, v):
                tau_b.setdefault((u,v), 1.0)

        cand = build_candidate_lists(H, L=L, dmd=dmd, theta_osnr=18.0, srlg_primary=None)
        # iteraciones ACO
        for _ in range(it_max):
            ants = max(5, H.number_of_nodes() // 2)
            candidates: List[List[str]] = []
            for _a in range(ants):
                p = construct_path_with_candidates(H, dmd, cand, alpha, beta, tau_r, tau_b)
                if p and feasible_RS_RSA_OSNR(H, p, dmd, theta_osnr=18.0):
                    candidates.append(p)
            # evaporación
            for e in tau_r: tau_r[e] *= (1 - rho)
            for e in tau_b: tau_b[e] *= (1 - rho)

            if candidates:
                # mejor P’ por utilidad coste = w2*delay + w3*hops + w4*slots_consumidos
                for p in candidates:
                    delay = sum(H[p[i]][p[i+1]]['delay'] for i in range(len(p)-1))
                    hops  = len(p)-1
                    # slots consumidos estimados ≈ bw
                    cost = 0.6*delay + 0.3*hops + 0.1*dmd.bw
                    if cost < best_cost:
                        best_cost = cost; best_path = p
                # refuerzo
                for i in range(len(best_path)-1):
                    e = (best_path[i], best_path[i+1])
                    tau_r[e] = tau_r.get(e,1.0) + 1.0 / max(1.0, best_cost)
                    if e in tau_b:
                        tau_b[e] = tau_b.get(e,1.0) + 0.5 / max(1.0, best_cost)

            if (time.time() - start_t)*1000 >= time_budget_ms:
                break

        if best_path: break
        k += 1  # expandir vecindario

    return best_path

# -------------------------------
# Ejemplo de uso (stub)
# -------------------------------
if __name__ == "__main__":
    G = nx.Graph()
    # construir pequeña topología de ejemplo
    for n in "ABCDEFGH":
        G.add_node(n)
    edges = [
        ("A","B", 1.0), ("B","C", 1.0), ("C","D", 1.2),
        ("A","E", 1.3), ("E","F", 1.0), ("F","G", 1.0), ("G","D", 1.2),
        ("B","F", 1.1), ("C","G", 1.1), ("E","B", 1.0), ("G","H", 1.5), ("H","D", 1.1)
    ]
    for u,v,delay in edges:
        G.add_edge(u,v, delay=delay, dist=80.0, osnr=20.0, srlg=1, 
                   slots=[1]*320, tau_r=1.0, tau_b=1.0)
    failed = ("B","C")

    dmd = Demand(s="A", d="D", bw=8, cls="gold", delay_max=10.0, hops_max=8)
    p_prime = ACO_neighborhood(G, dmd, failed_edge=failed, time_budget_ms=30)
    print("P’ encontrada:", p_prime)
