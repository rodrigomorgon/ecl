# Evolutionary Concept Learning using Examples with Categorical Attributes

from math import *
from random import *
from string import *
from time import *
import os
import sys

# Global variables

fs = {0:[], 1:['not'], 2:['and','or']} # function set
fd = {}                                # features domains
at = []                                # attributes names

# Evolutionary search

def ecl(problem,runs=1,save=True):
    (elapsed,hypothesis) = training(problem,runs)
    validation(problem,runs,elapsed,hypothesis,save)

def training(problem,runs,pop_size=25,generations=5000):
    ts = get_data(problem+'.ts.arff')
    (bt,(bf,bs)) = ('',(0,0))
    start = clock()
    for r in range(1,runs+1):
        print '\n%s [%d/%d]' % (problem,r,runs)
        run_start = clock()
        population = first_generation(pop_size,ts)
        (tree,(fitness,size)) = population[0]
        g = 1
        while g<generations and fitness<1:
            population = next_generation(population,pop_size,ts)
            if population[0][1][0]>fitness:
                elapsed = clock()-run_start
                (tree,(fitness,size)) = population[0]
                print 'Searching [%7.3f, %4d, %3d, %3d, %2d,  %.3f, %4d]' % (elapsed,g,pop_size,size,height(tree),fitness,error(tree,ts))
            g += 1
        if (fitness>bf) or (fitness==bf and size<bs): (bt,(bf,bs)) = (tree,(fitness,size))
        elapsed = clock()-run_start
        print 'Time....: %.2fs (%d generations)' % (elapsed, g)
    elapsed = clock()-start
    print '\n-----------------------------------------------------'
    print 'Total training time: %.1fs' % (clock()-start)
    print 'Best fitness.......: %.3f' % bf
    print 'Best hypothesis....: %s' % infix(bt)
    return (elapsed,bt)


def validation(problem,runs,elapsed,hypothesis,save):
    vs = get_data(problem+'.vs.arff')
    (tp,fp,tn,fn) = confusion_matrix(hypothesis,vs)
    Acc = acc(tp,fp,tn,fn)
    G_m = gmh(tp,fp,tn,fn)
    F_m = pfm(tp,fp,tn,fn)
    TPR = tpr(tp,fp,tn,fn)
    TNR = tnr(tp,fp,tn,fn)
    args = (runs,elapsed,Acc*100,G_m,F_m,TPR,TNR)
    print '\n====================================================='
    print 'Runs   Time  Accuracy G-measure F-measure   TPR   TNR'
    print '%4d %6.1f  %6.1f %% %9.3f %9.3f %5.3f %5.3f' % args
    print '-----------------------------------------------------\n'
    print '              Positive  Negative <-- classified as'
    print '            +-------------------'
    print '   Positive | %5d' % tp,
    print >> sys.stderr, '%9d' % fn
    print '  Negative | ',
    print >> sys.stderr, '%5d' % fp,
    print '%9d' % tn
    print '\n=====================================================\n'
    if save:
        f = open('%s.st.txt' % problem,'a')
        #if runs == 1: f.write('# Runs   Time  Accuracy  G-measure  F-measure    TPR    TNR\n\n')
        f.write('  %4d %6.1f  %8.1f  %9.3f  %9.3f  %5.3f  %5.3f\n' % args)
        f.close()

def get_data(filename):
    '''Get training or test data set'''
    global at, fd
    attributes = []
    data = []
    at = []
    fd = {}
    for line in open(filename):
        tokens = [token for token in line.replace('\n','').replace(',',' ').split(' ') if token != '']
        if tokens==[] or tokens[0] in ['%','@relation','@data']: continue
        if tokens[0]=='@attribute':
            attribute = tokens[1].replace('-','_')
            domain = [ v.replace('{','').replace('}','').replace('-','_') for v in tokens[2:]]
            at.append(attribute)
            fd[attribute]=tuple(domain)
            attributes.extend([attribute+'=='+"'"+v+"'" for v in domain])
        else:
            values = [v.replace('-','_') for v in tokens]
            data.append((tuple(values[:-1]),boolean(values[-1])))
    fs[0] = attributes[:-2]
    at = at[:-1]
    return data

def boolean(x):
    return int(x in ['1','true','yes','positive'])

def first_generation(pop_size,ts):
    '''Return a random population of distinct trees'''
    h = ceil(log(len(fs[0]),2))
    while number_of_trees(h)<2*pop_size: h += 1
    p = {}
    while len(p)<2*pop_size:
        t = random_tree(h)
        if t not in p: p[t] = (fitness(t,ts),size(t))
    population = sorted(p.items(), key = lambda x: x[1][0], reverse=True)
    bf = population[0][1][0]
    bs = float(population[0][1][1])
    return sorted(population, key = lambda x: x[1][0]*weight(x[1],bf,bs), reverse=True) 

def next_generation(population,pop_size,ts):
    '''Return a mutant population of distinct trees'''
    p = dict(population)
    q = {}
    for (t,_) in population:
        m = mutation(t)
        while m in p or m in q: m = mutation(m)
        q[m] = (fitness(m,ts),size(m))
    bf = population[0][1][0]
    bs = float(population[0][1][1])
    population = p.items()+q.items()
    return sorted(population, key = lambda x: x[1][0]*weight(x[1],bf,bs), reverse=True)[:pop_size]

# Random generation of expression trees

def number_of_trees(height):
    '''Return the number of expression trees of bounded height'''
    if height==1: return len(fs[0])
    k = number_of_trees(height-1)
    return len(fs[0]) + k*len(fs[1]) + k**2*len(fs[2])

def random_tree(height):
    '''Return a random expression tree of bounded height'''
    m = float(number_of_trees(height))
    if height==1 or random()<len(fs[0])/m: return choice(fs[0])
    n = float(number_of_trees(height-1))
    if random()<(n*len(fs[1]))/(m-len(fs[0])): return (choice(fs[1]),random_tree(height-1))
    return (choice(fs[2]),random_tree(height-1),random_tree(height-1))

def mutation(tree):
    '''Return a mutation of a tree'''
    n = random()
    if isinstance(tree,str) or                 n<0.01: return random_tree(2)                                           # 1
    elif tree[0] in fs[1] and len(fs[1])>1 and n<0.02: return (choice(fs[1]),tree[1])                                  # 1
    elif tree[0] in fs[1] and len(fs[1])>1 and n<0.03: return (choice(fs[1]),tree)                                     # 1
    elif tree[0] in fs[1] and                  n<0.23: return (tree[0],mutation(tree[1]))                              # 20
    elif tree[0] in fs[2] and                  n<0.24: return (choice(fs[2]),random_tree(2),tree[int(random()<0.5)+1]) # 1
    elif tree[0] in fs[2] and                  n<0.25: return (choice(fs[2]),random_tree(2),tree)                      # 1 
    elif tree[0] in fs[2] and len(fs[2])>1 and n<0.50: return (choice(fs[2]),tree[1],tree[2])                          # 25
    elif tree[0] in fs[2] and                  n<0.75: return (tree[0],mutation(tree[1]),tree[2])                      # 25
    elif tree[0] in fs[2] and                  n<1.00: return (tree[0],tree[1],mutation(tree[2]))# 25
    else: return mutation(tree)

# Evaluation of trees

def infix(tree):
    '''Return the infix form of a tree'''
    if isinstance(tree,str): return tree
    if tree[0] in fs[1]:
        s = infix(tree[1])
        if s[0]!='(': return tree[0]+'('+s+')'
        else: return tree[0]+s
    return '('+join([infix(argument) for argument in tree[1:]],' '+tree[0]+' ')+')'

def size(tree):
    '''Return the size of a tree'''
    if type(tree)==str: return 1
    return 1+sum([size(s) for s in tree[1:]])

def height(tree):
    '''Return the height of a tree'''
    if isinstance(tree,str): return 1
    return max([height(t) for t in tree[1:]])+1

def error(tree,ts):
    '''Return the error of a tree'''
    f = eval('lambda ' + ', '.join(at) + ':' + infix(tree))
    return sum([(int(f(*x))-int(y))**2 for (x,y) in ts])

def confusion_matrix(tree,ds):
    '''Return confusion matrix for a tree and a dataset'''
    global at
    f = eval('lambda ' + ', '.join(at) + ':' + infix(tree))
    c = {(0,0):0.0, (0,1):0.0, (1,0):0.0, (1,1):0.0}
    r = [(int(f(*x)),y) for (x,y) in ds]
    for p in r: c[p] += 1
    tp = c[(1,1)]
    fp = c[(1,0)]
    tn = c[(0,0)]
    fn = c[(0,1)]
    return (tp,fp,tn,fn)

def fitness(tree,ts):
    cm = confusion_matrix(tree,ts)
    return gmh(*cm)

def gmh(tp,fp,tn,fn): return sqrt(tpr(tp,fp,tn,fn)*tnr(tp,fp,tn,fn)) # Geometric mean heuristic
def gm(tpr,tnr): return sqrt(tpr*tnr)                                # Geometric mean function

# f-bf => 0.xxx
# (s-bs)*1e-6 => 0.000yyy
# v = 0.xxxyyy

def weight(x,bf,bs):
    f = float(x[0])
    s = float(x[1])
    v = 1 + (f-bf) + (bs-s)*1e-6
    if f>bf  and s< bs: return v+0.004
    if f>bf  and s==bs: return v+0.003
    if f>bf  and s> bs: return v+0.002
    if f==bf and s< bs: return v+0.001
    if f==bf and s==bs: return v
    if f<bf  and s< bs: return v-0.001  
    if f<bf  and s==bs: return v-0.002
    if f==bf and s> bs: return v-0.003
    if f<bf  and s> bs: return v-0.004



# Weka/ECL Confusion Matrix and Measures
#
#  TP | FN
# ----+----
#  FP | TN
# 
# Weka | TP Rate |  FP Rate | Precision | Recall | F-measure
# -----+---------+----------+-----------+--------+----------
# ECL  |   TPR   |    FNR   |    PPV    |  TPR   |    PFM
#      |   TNR   |    FPR   |    NPV    |  TNR   |    NFM

def acc(tp,fp,tn,fn): return (tp+tn)/(tp+fp+tn+fn)     # accuracy
def tpr(tp,fp,tn,fn): return tp/(tp+fn)                # positive rate
def tnr(tp,fp,tn,fn): return tn/(tn+fp)                # true negative rate
def fpr(tp,fp,tn,fn): return fp/(fp+tn)                # false positive rate 
def fnr(tp,fp,tn,fn): return fn/(tp+fn)                # false negative rate
def ppv(tp,fp,tn,fn): return tp/(tp+fp) if tp>0 else 0 # positive predictive value
def npv(tp,fp,tn,fn): return tn/(tn+fn) if tn>0 else 0 # negative predictive value
def pfm(tp,fp,tn,fn):                                  # positive F-measure 
    p = ppv(tp,fp,tn,fn)
    r = tpr(tp,fp,tn,fn)
    return 2*(p*r)/(p+r) if (p+r)>0 else 0
def nfm(tp,fp,tn,fn):                                  # negative F-measure
    p = npv(tp,fp,tn,fn)
    r = tnr(tp,fp,tn,fn)
    return 2*(p*r)/(p+r) if (p+r)>0 else 0

def problem_statistics(problem,f=sys.stdout):
    ts = get_data(problem+'.ts.arff')
    vs = get_data(problem+'.vs.arff')
    p = len([x for (x,y) in ts if y==1])
    n = len([x for (x,y) in ts if y==0])
    iss = prod([len(fd[a]) for a in at])
    f.write('# Instance space size: %5d (%d attributes, %d propositions)\n' % (iss,len(at),len(fs[0])))
    f.write('# Training set size..: %5d (%3.1f %%)\n' % (len(ts),100.0*len(ts)/iss))
    f.write('# Validation set size: %5d (%3.1f %%)\n' % (len(vs),100.0*len(vs)/iss))
    f.write('# Imbalance ratio....: %5d+ : %d- (%.1f %%)\n\n' % (p,n,100.0*p/(p+n)))

def prod(l):
    if len(l)==1: return l[0]
    return l[-1]*prod(l[:-1])

def experiments(problem,runs=1):
    f = open('%s.st.txt' % problem,'a')
    #problem_statistics(problem,f)
    f.write('# Runs   Time  Accuracy  G-measure  F-measure    TPR    TNR\n\n')
    f.close()
    for i in range(1,6):
        for r in range(1,runs+1):
            ecl(problem,1)

##experiments('./datasets/imbalance/imbalance-1') 
##experiments('./datasets/imbalance/imbalance-2') 
##experiments('./datasets/imbalance/imbalance-3') 
##experiments('./datasets/imbalance/imbalance-4') 
##experiments('./datasets/imbalance/imbalance-5')

##experiments('./datasets/interaction/credit') 
##experiments('./datasets/interaction/mutex-1') 
##experiments('./datasets/interaction/mutex-2')
##experiments('./datasets/interaction/symmetric-1') 
##experiments('./datasets/interaction/symmetric-2')
##
##experiments('./datasets/imbalance+interaction/imbalance+interaction-1')
##experiments('./datasets/imbalance+interaction/imbalance+interaction-2')
##experiments('./datasets/imbalance+interaction/imbalance+interaction-3')
##experiments('./datasets/imbalance+interaction/imbalance+interaction-4')
##experiments('./datasets/imbalance+interaction/imbalance+interaction-5') 
##
##experiments('./datasets/monks/monks-1')
##experiments('./datasets/monks/monks-2')
##experiments('./datasets/monks/monks-3')
##
##experiments('./datasets/natural/bankruptcy') 
##experiments('./datasets/natural/breast') 
##experiments('./datasets/natural/car')
##experiments('./datasets/natural/chess')
##experiments('./datasets/natural/flare')
##experiments('./datasets/natural/heart')
##experiments('./datasets/natural/housevotes')
##experiments('./datasets/natural/kr-vs-k-zero-one_vs_draw')
##experiments('./datasets/natural/lenses') 
##experiments('./datasets/natural/mushroom')
##experiments('./datasets/natural/nursery')       
##experiments('./datasets/natural/post-operative')
##experiments('./datasets/natural/tic-tac-toe')   
##experiments('./datasets/natural/zoo')
##
##experiments('./datasets/neutral/consumer-1')
##experiments('./datasets/neutral/consumer-2')
##experiments('./datasets/neutral/parity-1')
##experiments('./datasets/neutral/parity-2')
##

##experiments('./datasets/noise/noise-1')
##experiments('./datasets/noise/noise-2')
##experiments('./datasets/noise/noise-3')
##experiments('./datasets/noise/noise-4')
##experiments('./datasets/noise/noise-5')
