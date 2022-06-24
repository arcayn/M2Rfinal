import random
import itertools


def plucker(P, vl=None):
    #M1 = Matrix([[1,1],[0,1]])
    if vl is None: vl = list(P.vertices())
    M_ = Matrix([b for b in vl]).transpose()
    M = M_.right_kernel().matrix()
    rows,_ = M.dimensions()
    minors = M.minors(rows)
    return ProjectiveSpace(len(minors) - 1, ZZ)(minors)

def gen_all_plucker(dim):
    assert dim in (2, 3)
    for idx in range(15 if dim == 2 else 4318):
        pt = ReflexivePolytope(dim, idx)
        if pt.lattice().submodule(pt.vertices()).index_in(pt.lattice()) != 1:
            continue
        for vl in itertools.permutations(list(pt.vertices())):
            yield (plucker(pt, vl), idx)
    return

def graph_embedding(P):
    vl = list(P.vertices())
    edges = [f.vertices() for f in P.faces()[2]]
    edges = [(e[0], e[1]) for e in edges]
    weights = [abs(e[1] - e[0]) for e in edges]
    edges_anon = [sorted([vl.index(x), vl.index(y)]) for x,y in edges]

    return [list([(a,b,c) for a,b,c in vl]), sorted([(a[0], a[1], b) for a,b in zip(edges_anon, weights)])]

def flatten2d(l):
    r = []
    for ll in l: r += ll
    return r

def face_graph_embedding(P):
    faces = flatten2d(P.faces())
    adjacency = [[0 for _ in faces] for _ in faces]
    for i,f1 in enumerate(faces):
        for j,f2 in enumerate(faces):
            if f1.dim() == f2.dim() + 1 and f2 in flatten2d(f1.faces()):
                adjacency[i][j] = 1
                adjacency[j][i] = 1
    
    return adjacency

def vertex_matrix_embedding(P):
    vl = list(P.vertices())
    l = [list(v) + [0] * 13 for v in vl]
    for ray in P.faces()[-3]:
        v1,v2 = ray.vertices()
        l[vl.index(v1)][3+vl.index(v2)] = 1
        l[vl.index(v2)][3+vl.index(v1)] = 1

    return str(l)

def map_lattice_pt(pt, M, permute=True):
    V = [M * v for v in pt.vertices()]
    random.shuffle(V)
    return LatticePolytope(V)

def make_polytope_sample_train():
    r = []
    for x in range(4319):
        print (x)
        base_pt = ReflexivePolytope(3, x)
        if len(base_pt.vertices()) != 10:
            continue
        for _ in range(50):
            M = random_matrix(ZZ,3,3)
            while abs(M.det()) != 1:
                M = random_matrix(ZZ,3,3)
            pt = map_lattice_pt(base_pt, M)
            gpt = [vertex_matrix_embedding(pt), x]
            r.append(str(gpt) + "\n")
    
    with open("data/train_matrices.txt", "w") as f:
        f.writelines(r)

def make_polytope_sample_test():
    r = []
    import random
    for _ in range(2000):
        x = random.randint(0, 4318)
        base_pt = ReflexivePolytope(3, x)
        while len(base_pt.vertices()) != 10:
            x = random.randint(0, 4318)
            base_pt = ReflexivePolytope(3, x)
        M = random_matrix(ZZ,3,3)
        while abs(M.det()) != 1:
            M = random_matrix(ZZ,3,3)
        pt = map_lattice_pt(base_pt, M)
        gpt = [vertex_matrix_embedding(pt), x]
        r.append(str(gpt) + "\n")
    
    with open("data/test_matrices.txt", "w") as f:
        f.writelines(r)

make_polytope_sample_test()
make_polytope_sample_train()

def gen_whole_plucker_sample(dim):
    assert dim in (2, 3)
    r = []
    for idx in range(15 if dim == 2 else 4318):
        pt = ReflexivePolytope(dim, idx)
        if pt.lattice().submodule(pt.vertices()).index_in(pt.lattice()) != 1:
            continue
            
        rr = set()
        for vl in itertools.permutations(list(pt.vertices())):
            rr.add(plucker(pt, vl))
        r.append((rr,idx))
    return r

def gen_random_plucker_sample(dim):
    assert dim in (2, 3)

    r = []
    for idx in range(15 if dim == 2 else 4318):
        print (idx)
        pt = ReflexivePolytope(dim, idx)
        if pt.lattice().submodule(pt.vertices()).index_in(pt.lattice()) != 1:
            continue
        if len(pt.vertices()) < 5:
            continue
        
        rr = set()
        if len(pt.vertices()) <= 5:
            for vl in itertools.permutations(list(pt.vertices())):
                rr.add(plucker(pt, vl))
        else:
            print ("H")
            t = list(pt.vertices())
            random.shuffle(t)
            cnt = 0
            while len(rr) < 75:
                rr.add(plucker(pt, t))
                random.shuffle(t)
                cnt += 1
                if cnt > 1000:
                    print ("BAD", cnt)
                    break
        
        if len(rr) < 75:
            continue
        rl = random.sample(list(rr), 75)
        
        r.append((rl, idx))
    return r


def gen_reflexive_polytope(dim):
    # will work on generating arbitrary reflexive polytopes
    # but for now we focus on generating isomorphisms to the PALP db
    assert dim in (2, 3)

    idx = random.randint(0, 15 if dim == 2 else 4318)
    base_pt = ReflexivePolytope(dim, idx)
    """
    transform = [random.randint(0, 100) for _ in range(dim + 1)]

    pt = base_pt
    for e,M in zip(transform, GL(dim, ZZ).as_matrix_group().gens()):
        pt = map_lattice_pt(pt, (M ^ e)) 
    """

    M = random_matrix(ZZ,dim,dim)
    while abs(M.det()) != 1:
        M = random_matrix(ZZ,dim,dim)
    pt = map_lattice_pt(base_pt, M)
    
    # make sure we have actually performed a CoB
    assert pt.index() == idx
    return pt, idx

def gen_primitive_polytope(dim, n):
    assert n >= dim + 1


def write_plucker_sample(dim):
    sample = gen_whole_plucker_sample(dim)
    sample = [[[list(s) for s in S],i] for S,i in sample]
    print (sample)
    with open("plucker_s.txt", "wb") as f:
        f.write(str(sample).encode())

def plotPoly(P):
    rad = 3
    plot = [[' ' for _ in range(2 * rad + 1)] for _ in range(2 * rad + 1)]
    for x,y in P.vertices():
        plot[rad - y][rad + x] = '.'
    plot[rad][rad] = 'O'
    for pp in plot:
        print (''.join(pp))

plotPoly(ReflexivePolytope(2, 11))
plotPoly(ReflexivePolytope(2, 8))
for i in range(16):
    print (i, plucker(ReflexivePolytope(2, i)))