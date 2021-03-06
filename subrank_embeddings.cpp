// 
// This code allows the computation of subgraph embeddings. 
//
// It receives in input a file of the form:
// #i (this is the ith subgraph) n1 pr(n1|Si ) n2 pr(n2|Si) ... 
// where n1, n2 are nodes in subgraph Si and pr(n1| Si) is the pagerank of node n1 in subgraph Si
// and also a file containing the input graph. 
// It gives in output a file containing on each line i an embedding for the ith subgraph in the input file. 
// If the input file corresponds to ego networks, we suppose that the nodes in the original graph are numbered from 0 to num_nodes - 1
// and on each line i-1 we have the ego network of node i.
//
// This code is based on Subgraph: Subgraph Embeddings via a Proximity Measure
//
// Oana Balalau, Sagar Goyal
//
// The code is an adaptation of the code released with the VERSE paper (BERSE: Versatile Graph Embeddings from Similarity Measures).
//  
//

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>
#include <string>
#include <sstream>
#include <map>

using namespace std;

#if defined(__AVX2__) || \
defined(__FMA__)
#define VECTORIZE 1
#define AVX_LOOP _Pragma("omp simd")
#else
#define AVX_LOOP
#endif

#ifndef UINT64_C
#define UINT64_C(c) (c##ULL)
#endif

#define SIGMOID_BOUND 6.0
#define DEFAULT_ALIGN 128

typedef unsigned long long ull;

bool silent = false;
int n_threads = 1;
float global_lr = 0.0025f;
int n_epochs = 100000;
int n_hidden = 128;
int n_samples = 3;
float ppralpha = 0.85f;



std::map<int, std::vector<std::pair<int, float>>> pr_ego;

std::map<int, std::vector<std::pair<int, float>>> pr_ego_belong;


ull total_steps;
ull step = 0;

ull nv = 0, ne = 0;
int *offsets;
int *edges;

float *w0;

const int sigmoid_table_size = 1024;
float *sigmoid_table;
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);

uint64_t rng_seed[2];

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}


bool sortbysec(const pair<int,float> &a,
              const pair<int,float> &b)
{
    return (a.second > b.second);
}


uint64_t lrand() {
  const uint64_t s0 = rng_seed[0];
  uint64_t s1 = rng_seed[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  rng_seed[1] = rotl(s1, 36);                   // c
  return result;
}

static inline double drand() {
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

inline void *aligned_malloc(
  size_t size,
  size_t align) {
#ifndef _MSC_VER
void *result;
if (posix_memalign(&result, align, size)) result = 0;
#else
void *result = _aligned_malloc(size, align);
#endif
return result;
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
_aligned_free(ptr);
#else
free(ptr);
#endif
}

void init_sigmoid_table()
{
  float x;
  sigmoid_table = static_cast<float *>(aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  for (int k = 0; k != sigmoid_table_size; k++)
  {
    x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float FastSigmoid(float x)
{
  if (x > SIGMOID_BOUND)
    return 1;
  else if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

inline int irand(int min, int max) { return lrand() % (max - min) + min; }

inline int irand(int max) { return lrand() % max; }



inline int sample_neighbor(int node) {
  if (offsets[node] == offsets[node + 1])
    return -1;
  return edges[irand(offsets[node], offsets[node + 1])];
}

// In this function a random walker will continue a walk with probability ppralpha
// the next node in the walk is chosen with uniform probability from the outneighbors
// of the current node. 
 
inline int sample_rw(int node)
 {
   int n2 = node;
   while (drand() < ppralpha)
   {
     int neighbor = sample_neighbor(n2);
     if (neighbor == -1)
       return n2;
     n2 = neighbor;
   }
   return n2;
 }


// In this function we sample a node from the distribution distr
//
// This distribution corresponds to the pagerank vector in subgraph
//  


inline int sample_distr(int node,  vector<pair<int, float> > distr)
{
  float rando = drand();
  double run_sum = 0.0;

  int ans = node;
  int flag = 0 ;
  for ( vector <pair<int,float> >::const_iterator it = distr.begin() ; it != distr.end(); it++)
  {
        run_sum += (double)it->second;
        if(rando < run_sum)
        {
          ans = it->first;
          flag=1;
          break;
        }

  }
  if(flag==1)
  {
    return ans;
  }
  else return node;

}


// In this function we start from an initial subgraph "first_subgraph", which is the index of a subgraph S_node
// we find a new node in that subgraph with probability S_node (start_node)
// from that node we do a random walk in the graph and we arrive to node inter_node
// We find to which subgraph it is more likely that the node inter_node belongs, using the probability pr_ego_belong
// and we return this subgraph "second subgraph" 
inline int sample_pair(int first_subgraph)
{
  int start_node = sample_distr(first_subgraph, pr_ego[first_subgraph]);
  int inter_node = sample_rw(start_node);
  int second_subgraph = sample_distr(inter_node, pr_ego_belong[inter_node]);
  return second_subgraph;

}
int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        cout << "Argument missing for " << str << endl;
        exit(1);
      }
      return a;
    }
  return -1;
}

inline void update(float *w_s, float *w_t, int label, const float bias)
{
  float score = -bias;
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  score = (label - FastSigmoid(score)) * global_lr;

AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t[c] += score * w_s[c];

AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c];
}

void Train()
{
#pragma omp parallel num_threads(n_threads)
  {
    const float nce_bias = log(nv);
    const float nce_bias_neg = log(nv / float(n_samples));
    int tid = omp_get_thread_num();
    ull last_ncount = 0;
    ull ncount = 0;
    float lr = global_lr;
#pragma omp barrier
    while (1)
    {
      if (ncount - last_ncount > 10000)
      {
        ull diff = ncount - last_ncount;
        #pragma omp atomic
        step += diff;
        if (step > total_steps)
          break;
        if (tid == 0)
          if (!silent)
            cout << fixed << "\r Progress " << std::setprecision(2)
                 << step / (float)(total_steps + 1) * 100 << "%";
        last_ncount = ncount;
      }


      size_t n1 = irand(nv);
      size_t n2 = sample_pair(n1);
      update(&w0[n1 * n_hidden], &w0[n2 * n_hidden], 1, nce_bias);
      for (int i = 0; i < n_samples; i++)
      {
        size_t neg = irand(nv);
        update(&w0[n1 * n_hidden], &w0[neg * n_hidden], 0, nce_bias_neg);
      }
      ncount++;
    }
  }
}



int main(int argc, char **argv) {
  int a = 0;
  string network_file, embedding_file, pr_file;
  ull x = time(nullptr);
  for (int i = 0; i < 2; i++)
  {
      ull z = x += UINT64_C(0x9E3779B97F4A7C15);
      z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
      z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
      rng_seed[i] = z ^ z >> 31;
  }

  init_sigmoid_table();

  if ((a = ArgPos(const_cast<char *>("-graph"), argc, argv)) > 0)
    network_file = argv[a + 1];
  else {
    cout << "Graph file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }
  if ((a = ArgPos(const_cast<char *>("-embedding"), argc, argv)) > 0)
    embedding_file = argv[a + 1];
  else {
    cout << "Output file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }
  if ((a = ArgPos(const_cast<char *>("-prfile"), argc, argv)) > 0)
    pr_file = argv[a + 1];
  else {
    cout << "PR file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }

  if ((a = ArgPos(const_cast<char *>("-dim"), argc, argv)) > 0)
    n_hidden = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-silent"), argc, argv)) > 0)
    silent = true;
  if ((a = ArgPos(const_cast<char *>("-nsamples"), argc, argv)) > 0)
    n_samples = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-threads"), argc, argv)) > 0)
    n_threads = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-steps"), argc, argv)) > 0)
    n_epochs = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-lr"), argc, argv)) > 0)
    global_lr = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-alpha"), argc, argv)) > 0)
    ppralpha = atof(argv[a + 1]);

  ifstream embFile(network_file, ios::in | ios::binary);

  if (embFile.is_open())
  {
    char header[] = "----";
    embFile.seekg(0, ios::beg);
    embFile.read(header, 4);
    if (strcmp(header, "XGFS") != 0)
    {
      cout << "Invalid header!: " << header << endl;
      return 1;
    }
    embFile.read(reinterpret_cast<char *>(&nv), sizeof(long long));
    embFile.read(reinterpret_cast<char *>(&ne), sizeof(long long));
    offsets = static_cast<int *>(aligned_malloc((nv + 1) * sizeof(int32_t), DEFAULT_ALIGN));
    edges = static_cast<int *>(aligned_malloc(ne * sizeof(int32_t), DEFAULT_ALIGN));
    embFile.read(reinterpret_cast<char *>(offsets), nv * sizeof(int32_t));
    offsets[nv] = (int)ne;
    embFile.read(reinterpret_cast<char *>(edges), sizeof(int32_t) * ne);
    cout << "nv: " << nv << ", ne: " << ne << endl;
    embFile.close();
  }
  else
  {
    return 0;
  }


  // PR file read --------------------------------------------- ----------------------------


  std::fstream in(pr_file);
  std::string line;
  int veci = 0;

  while(std::getline(in,line))
  {
  	std::stringstream sr(line);
  	int index = 0 ;
  	sr>>index;
  	std::map<int, float> pr_values;
  	std::vector<std::pair<int, float>> pr_values_list;
  	float value = 0.0;
  	int neigh = 0;
  	while (sr >> neigh && sr >> value)
  	{	 
  		pr_values[neigh] = value;
   		pr_values_list.push_back(std::make_pair(neigh, value));
   		if (pr_ego_belong.count(neigh) > 0 )
   		{
   		pr_ego_belong[neigh].push_back(std::make_pair(index, value));
   		}
   		else
   		{
   		std::vector<std::pair<int, float>> pr_list;
   		pr_list.push_back(std::make_pair(index, value));
   		pr_ego_belong[neigh] = pr_list;
   		}
  	}
  	pr_ego[index] = pr_values_list;
  }

  for (std::map<int,  std::vector<std::pair<int, float>> >::iterator it=pr_ego_belong.begin(); it!=pr_ego_belong.end(); ++it)
  {   std::vector<std::pair<int, float>> to_sort = it->second;
      sort(to_sort.begin(), to_sort.end(), sortbysec);
      float all_prob = 0.0;
      for ( vector < std::pair<int, float> >::iterator i = to_sort.begin() ; i != to_sort.end(); i++){
           all_prob += i->second;
        }
        if (all_prob == 0)
           continue;
        for ( vector < std::pair<int, float> >::iterator i = to_sort.begin() ; i != to_sort.end(); i++){
           i->second = i->second/all_prob;
        }
       it->second = to_sort;
  }
  w0 = static_cast<float *>(aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  // random initialisation
  for (size_t i = 0; i < nv * n_hidden; i++)
    w0[i] = drand() - 0.5;

  total_steps = n_epochs * (long long)nv;
  cout << "Total steps (mil): " << total_steps / 1000000. << endl;
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  // Now all the things will start happening
  Train();
  chrono::steady_clock::time_point end = chrono::steady_clock::now();

  cout << endl
       << "Calculations took "
       << chrono::duration_cast<std::chrono::duration<float>>(end - begin).count()
       << " s to run" << endl;


  for (size_t i = 0; i < nv * n_hidden; i++)
    if (w0[0] != w0[0])
    {
      cout << endl << "NaN! Not saving the result.." << endl;
      return 1;
    }


  ofstream output(embedding_file, std::ios::binary);
  output.write(reinterpret_cast<char *>(w0), sizeof(float) * n_hidden * nv);
  output.close();
}
