% neighborhood_khop.m
function H = neighborhood_khop(G, s, d, failed_edge, k)
% G: struct con campos G.A (adj matrix), G.delay, G.osnr, G.slots{u,v}, G.srlg
% s,d: índices 1..N
% failed_edge: [u v]
    A = G.A;
    A(failed_edge(1), failed_edge(2)) = 0;
    A(failed_edge(2), failed_edge(1)) = 0;

    Ns = bfs_khop(A, s, k);
    Nd = bfs_khop(A, d, k);
    nodes = union(Ns, Nd);
    if isempty(nodes)
        H = G; H.A = A; return;
    end
    idx = false(1,size(A,1)); idx(nodes) = true;
    H = subgraph_from_mask(G, idx);  % construye subgrafo (A, delay, osnr, slots, srlg)
end

% bfs_khop.m
function nodes = bfs_khop(A, s, k)
    N = size(A,1);
    visited = false(1,N); visited(s) = true;
    frontier = s; depth = 0;
    while ~isempty(frontier) && depth < k
        depth = depth + 1;
        next = [];
        for u = frontier
            v = find(A(u,:)>0);
            next = [next, v(~visited(v))]; %#ok<AGROW>
        end
        visited(next) = true; frontier = unique(next);
    end
    nodes = find(visited);
end

% build_candidate_lists.m
function cand = build_candidate_lists(H, L, bw, theta_osnr, srlg_primary)
    N = size(H.A,1);
    cand = cell(N,1);
    for u = 1:N
        neigh = find(H.A(u,:)>0);
        scores = zeros(1,numel(neigh));
        for k = 1:numel(neigh)
            v = neigh(k);
            scores(k) = eta(H, u, v, bw, theta_osnr, srlg_primary);
        end
        [~, idx] = sort(scores, 'descend');
        cand{u} = neigh(idx(1:min(L, numel(idx))));
    end
end

% eta.m
function val = eta(H, u, v, bw, theta_osnr, srlg_primary)
    delay = H.delay(u,v);
    osnr  = H.osnr(u,v);
    slots = H.slots{u,v}; % vector 0/1
    contig = max_contig_ones(slots);
    lam = 1.0;
    if ~isempty(srlg_primary) && H.srlg(u,v) == srlg_primary
        lam = 0.5;
    end
    osnr_fac = 1/(1 + exp(-(osnr - theta_osnr)));
    val = (1/max(1e-9,delay)) * osnr_fac * (1 + contig/numel(slots)) * lam;
end

% max_contig_ones.m
function m = max_contig_ones(bits)
    m=0; c=0;
    for i=1:numel(bits)
        if bits(i)==1, c=c+1; m=max(m,c); else, c=0; end
    end
end

% feasible_RS_RSA_OSNR.m
function ok = feasible_RS_RSA_OSNR(H, path, bw, delay_max, hops_max, theta_osnr)
    ok = false;
    if numel(path) < 2, return; end
    % delay y hops
    d=0;
    for i=1:numel(path)-1
        d = d + H.delay(path(i), path(i+1));
    end
    if d > delay_max || (numel(path)-1) > hops_max, return; end
    % continuidad/contigüidad
    S = [];
    for i=1:numel(path)-1
        arr = H.slots{path(i), path(i+1)};
        if isempty(S), S = arr; else, S = S & arr; end
    end
    if max_contig_ones(S) < bw, return; end
    % OSNR
    min_osnr = inf;
    for i=1:numel(path)-1
        min_osnr = min(min_osnr, H.osnr(path(i), path(i+1)));
    end
    if min_osnr < theta_osnr, return; end
    ok = true;
end

% acoN.m (bucle principal simplificado)
function best = acoN(G, dmd, failed_edge, params)
    alpha=params.alpha; beta=params.beta; rho=params.rho;
    L=params.L; k=params.k_init; kmax=params.k_max;
    itmax=params.it_max; budget=params.time_budget_ms;

    t0 = tic; best=[]; best_cost=inf;
    tau_r = containers.Map('KeyType','char','ValueType','double');
    tau_b = containers.Map('KeyType','char','ValueType','double');

    while (toc(t0)*1000 < budget) && (k <= kmax)
        H = neighborhood_khop(G, dmd.s, dmd.d, failed_edge, k);
        cand = build_candidate_lists(H, L, dmd.bw, params.theta_osnr, []);
        for it=1:itmax
            ants = max(5, round(size(H.A,1)/2));
            for a=1:ants
                p = construct_with_candidates(H, dmd, cand, alpha, beta, tau_r, tau_b, params);
                if ~isempty(p) && feasible_RS_RSA_OSNR(H, p, dmd.bw, dmd.delay_max, dmd.hops_max, params.theta_osnr)
                    cost = path_cost(H, p, dmd);
                    if cost < best_cost, best=p; best_cost=cost; end
                end
            end
            % evaporación (τ ← (1-ρ)τ)
            keys = tau_r.keys; for i=1:numel(keys), tau_r(keys{i}) = (1-rho)*tau_r(keys{i}); end
            keys = tau_b.keys; for i=1:numel(keys), tau_b(keys{i}) = (1-rho)*tau_b(keys{i}); end
            if toc(t0)*1000 >= budget, break; end
        end
        if ~isempty(best), break; end
        k = k + 1;
    end
end
