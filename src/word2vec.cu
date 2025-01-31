#include <cstdio>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <cuda_runtime.h>
#include "type.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <unistd.h>
#include "edge_container.hpp"
#include "lr_scheduler.hpp"
#include <algorithm>

using std::vector;
using std::string;
using std::cout;
using std::endl;

#define DELTA_R  100
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

#define MAX_SENTENCE 15000
#define checkCUDAerr(err) {\
  cudaError_t cet = err;\
  if (cudaSuccess != cet) {\
    printf("%s %d : %s\n", __FILE__, __LINE__, cudaGetErrorString(cet));\
    exit(0);\
  }\
}


const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

int my_rank;
float *last_emb;

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, min_reduce = 1, reuseNeg = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
float alpha = 0.025, starting_alpha, sample = 1e-3;
float *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

// FOR CUDA
int *vocab_codelen, *vocab_point, *d_vocab_codelen, *d_vocab_point;
char *vocab_code, *d_vocab_code;
int *d_table;
float *d_syn0, *d_syn1, *d_expTable;

__device__ float reduceInWarp(float f) {
  for (int i=warpSize/2; i>0; i/=2) {
    f += __shfl_sync(0xFFFFFFFF, f, i, 32);
  }
  return f;
}

__device__ void warpReduce(volatile float* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

// calculate cosine similarity of all nodes themselves
__global__ void cosine_similarity_kernel(float *d_vectors, float *d_result, long long  v, int dim){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("thread [%d,%d]\n",i,j);
	if( i < v && j < v) {
		float dot_product = 0.0f;
		float norm_i = 0.0f;
		float norm_j = 0.0f;

		// calculate dot_product and L2 norm.
		for (int k = 0; k < dim; ++k) {
			float vec_i = d_vectors[i * dim + k];
      float vec_j = d_vectors[j * dim + k];
			dot_product += vec_i * vec_j;
			norm_i += vec_i * vec_i;
			norm_j += vec_j * vec_j;	
		}
		norm_i = sqrt(norm_i);
		norm_j = sqrt(norm_j);
		// calculate similarity
		if(norm_i > 0.0f && norm_j > 0.0f) {
			d_result[i * v + j] = dot_product / (norm_i * norm_j);
		}else {
			d_result[i * v + j] = 0.0f; // prevent division by zero
		}
	}

}

__device__ float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}


__global__ void compute_kl_divergence_kernel(float *d_A,float *d_B, float *d_result, long long  v){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( idx < v * v) {
		float p = sigmoid(d_A[idx]);
		float q = sigmoid(d_B[idx]);
    // printf("sigmid(%d): %f, %f\n",idx,p,q);
		// 计算相对熵的部分贡献
		if(p > 0.0f && q > 0.0f){
			float contribution = p * logf(p /q );
      // printf("compute kl idx: %d val: %.2f\n",idx, contribution);
			atomicAdd(d_result, contribution); //  sum up
		}
	}
}

void  compute_kl_node_and_emb(float *h_node_sim, float *h_emb,float* h_result, long long  v,int dim){
	float *d_node_sim, *d_emb, *d_cosine, *d_result;
	cudaMalloc((void**)&d_emb, v * dim * sizeof(float));	
	cudaMalloc((void**)&d_node_sim, v * v * sizeof(float));	
	cudaMalloc((void**)&d_cosine, v * v * sizeof(float));	
	cudaMalloc((void**)&d_result, sizeof(float));
	
	cudaMemset(d_result, 0, sizeof(float));

	cudaMemcpy(d_node_sim,h_node_sim, v * v * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_emb,h_emb, v * dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(256);
	dim3 gridSize((v*v + blockSize.x - 1) / blockSize.x);

	// d_cosine 
	cosine_similarity_kernel<<<gridSize, blockSize>>>(d_emb,d_cosine,v,dim);

	// kl_divergence
	compute_kl_divergence_kernel<<<gridSize,blockSize>>>(d_node_sim,d_cosine,d_result,v);

	// copy result from device to host
	cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_node_sim);
	cudaFree(d_emb);
	cudaFree(d_cosine);
	cudaFree(d_result);
}
void write2file(float* ptr,size_t size,char* filename){
	FILE* f = fopen(filename,"w");
	if(f==NULL){
		printf("Failed to open %s\n",filename);
		return;
	}
	for(size_t i = 0;i<size; i++){
		fprintf(f,"%.3f ",ptr[i]);
	}
	fclose(f);
}

void  compute_kl_from_emb(float *emb1, float *emb2,float* h_result, long long v,int dim){
	float *d_emb1, *d_emb2, *d_cosine1, *d_cosine2, *d_result;
	cudaMalloc((void**)&d_emb1, v * dim * sizeof(float));	
	cudaMalloc((void**)&d_emb2, v * dim * sizeof(float));	
	cudaMalloc((void**)&d_cosine1, (size_t)v * v * sizeof(float));	
	cudaMalloc((void**)&d_cosine2, (size_t)v * v * sizeof(float));	
	cudaMalloc((void**)&d_result, sizeof(float));
	
	cudaMemset(d_result, 0, sizeof(float));


	cudaMemcpy(d_emb1,emb1, v * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_emb2,emb2, v * dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(16,16);
	dim3 gridSize((v+blockSize.x-1) / blockSize.x,(v+blockSize.y-1)/blockSize.y);

	// d_cosine 
	cosine_similarity_kernel<<<gridSize, blockSize>>>(d_emb1,d_cosine1,v,dim);
	cosine_similarity_kernel<<<gridSize, blockSize>>>(d_emb2,d_cosine2,v,dim);

	float *h_cosine = (float*)malloc( (size_t)v * v * sizeof(float));
	if(h_cosine == NULL)printf("Failed to allocate Mem\n");
  cudaDeviceSynchronize();
	cudaMemcpy(h_cosine,d_cosine1, (size_t)v * v * sizeof(float), cudaMemcpyDeviceToHost);
	write2file(h_cosine,(size_t)v * v,"cosine1.txt");
	cudaMemcpy(h_cosine,d_cosine2, (size_t)v * v  * sizeof(float), cudaMemcpyDeviceToHost);
	write2file(h_cosine,(size_t)v* v,"cosine2.txt");

	int kl_blockSize = 256;
  int kl_gridSize = (v * v + kl_blockSize - 1) / kl_blockSize;
	// kl_divergence
	compute_kl_divergence_kernel<<<kl_gridSize,kl_blockSize>>>(d_cosine1,d_cosine2,d_result,v);
  cudaDeviceSynchronize();

	// copy result from device to host
	cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_emb1);
	cudaFree(d_emb2);
	cudaFree(d_cosine1);
	cudaFree(d_cosine2);
	cudaFree(d_result);
}


template<unsigned int VSIZE>
__global__ void __sgNegReuse(const int window, const int layer1_size, const int negative, const int vocab_size, float alpha,
    const int* __restrict__ sen, const int* __restrict__ sentence_length,
    float *syn1, float *syn0, const int *negSample)
{
  __shared__ float neu1e[VSIZE];

  const int sentIdx_s = sentence_length[blockIdx.x];
  const int sentIdx_e = sentence_length[blockIdx.x + 1];
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int dxy = blockDim.x * blockDim.y;

  int _negSample;
  if (threadIdx.y < negative) {                                         // Get the negative sample
    _negSample = negSample[blockIdx.x * negative + threadIdx.y];
  }

  for (int sentPos = sentIdx_s; sentPos < sentIdx_e; sentPos++) {
    int word = sen[sentPos];                                            // Target word
    if (word == -1) continue;

    for (int a=0; a<window*2+1; a++) if (a != window) {
      int c = sentPos - window + a;                                     // The index of context word
      if (c >= sentIdx_s && c < sentIdx_e && sen[c] != -1) {
        int l1 = sen[c] * layer1_size;

        for (int i=tid; i<layer1_size; i+=dxy) {
          neu1e[i] = 0;
        }
        __syncthreads();

        int target, label, l2;
        float f = 0, g;
        if (threadIdx.y == negative) {                                  // Positive sample
          target = word;
          label = 1;
        } else {                                                        // Negative samples
          if (_negSample == word) goto NEGOUT;
          target = _negSample;
          label = 0;
        }
        l2 = target * layer1_size;

        for (int i=threadIdx.x; i<layer1_size; i+=blockDim.x) {         // Get gradient
          f += syn0[i + l1] * syn1[i + l2];
        }
        f = reduceInWarp(f);
        if      (f >  MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else {
          int tInt = (int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
          float t = exp((tInt / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
          t = t / (t + 1);
          g = (label - t) * alpha;
        }
        
        for (int i=threadIdx.x; i<layer1_size; i+=warpSize) {
          atomicAdd(&neu1e[i], g * syn1[i + l2]);
        }
        for (int i=threadIdx.x; i<layer1_size; i+=warpSize) {           // Update syn1 of negative sample
          syn1[i + l2] += g * syn0[i + l1];
        }

NEGOUT:
        __syncthreads();

        for (int i=tid; i<layer1_size; i+=dxy) {                        // Update syn0 of context word
          atomicAdd(&syn0[i + l1], neu1e[i]);
        }
      }
    }
  }
}

template<unsigned int FSIZE>
__global__ void skip_gram_kernel(int window, int layer1_size, int negative, int hs, int table_size, int vocab_size, float alpha,
    const float* __restrict__ expTable, const int* __restrict__ table, 
    const int* __restrict__ vocab_codelen, const int* __restrict__ vocab_point, const char* __restrict__ vocab_code,
    const int* __restrict__ sen, const int* __restrict__ sentence_length, float *syn1, float *syn0)
{
  __shared__ float f[FSIZE], g;

  int sent_idx_s = sentence_length[blockIdx.x];
  int sent_idx_e = sentence_length[blockIdx.x + 1]; 
  unsigned long next_random = blockIdx.x;

  if (threadIdx.x < layer1_size) for (int sentence_position = sent_idx_s; sentence_position < sent_idx_e; sentence_position++) {
    int word = sen[sentence_position];
    if (word == -1) continue;
    float neu1e = 0;
    next_random = next_random * (unsigned long)2514903917 + 11; 
    int b = next_random % window;

    for (int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      int c = sentence_position - window + a;
      if (c <  sent_idx_s) continue;
      if (c >= sent_idx_e) continue;
      int last_word = sen[c];
      if (last_word == -1) continue;
      int l1 = last_word * layer1_size;
      neu1e = 0;

      // HIERARCHICAL SOFTMAX
      if (hs) for (int d = vocab_codelen[word]; d < vocab_codelen[word+1]; d++) {
        int l2 = vocab_point[d] * layer1_size;

        if (threadIdx.x <  FSIZE) f[threadIdx.x] = syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
        __syncthreads();
        if (threadIdx.x >= FSIZE) f[threadIdx.x%(FSIZE)] += syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
        __syncthreads();
        for (int i=(FSIZE/2); i>0; i/=2) {
          if (threadIdx.x < i) f[threadIdx.x] += f[i + threadIdx.x];
          __syncthreads();
        }

        if      (f[0] <= -MAX_EXP) continue;
        else if (f[0] >=  MAX_EXP) continue;
        else if (threadIdx.x == 0) {
          f[0] = expTable[(int)((f[0] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (1 - vocab_code[d] - f[0]) * alpha;
        }
        __syncthreads();

        neu1e += g * syn1[threadIdx.x + l2];
        atomicAdd(&syn1[threadIdx.x + l2], g * syn0[threadIdx.x + l1]);
      }

      // NEGATIVE SAMPLING
      if (negative > 0) for (int d = 0; d < negative + 1; d++) {
        int target, label;
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long)25214903917 + 11; 
          target = table[(next_random >> 16) % table_size];
          if (target == 0)    target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        int l2 = target * layer1_size;

        if (threadIdx.x <  FSIZE) f[threadIdx.x] = syn0[threadIdx.x +l1] * syn1[threadIdx.x + l2];
        __syncthreads();
        if (threadIdx.x >= FSIZE) f[threadIdx.x%(FSIZE)] += syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
        __syncthreads();
        for (int i=(FSIZE/2); i>0; i/=2) {
          if (threadIdx.x < i)
            f[threadIdx.x] += f[i + threadIdx.x];
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          if (f[0] >  MAX_EXP)
            g = (label - 1) * alpha;
          else if (f[0] < -MAX_EXP)
            g = (label - 0) * alpha;
          else
            g = (label - expTable[(int)((f[0]+MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        }
        __syncthreads();

        neu1e += g * syn1[threadIdx.x + l2];
        atomicAdd(&syn1[threadIdx.x + l2], g * syn0[threadIdx.x + l1]);
      }

      atomicAdd(&syn0[threadIdx.x + l1], neu1e);
    }
  }
}

template<unsigned int FSIZE>
__global__ void cbow_kernel(int window, int layer1_size, int negative, int hs, int table_size, int vocab_size, float alpha,
    const float* __restrict__ expTable, const int* __restrict__ table,
    const int* __restrict__ vocab_codelen, const int* __restrict__ vocab_point, const char* __restrict__ vocab_code,
    const int* __restrict__ sen, const int* __restrict__ sentence_length, float *syn1, float *syn0)
{
  __shared__ float f[FSIZE], g;

  int sent_idx_s = sentence_length[blockIdx.x];
  int sent_idx_e = sentence_length[blockIdx.x + 1];
  unsigned long next_random = blockIdx.x;

  if (threadIdx.x < layer1_size) for (int sentence_position = sent_idx_s; sentence_position < sent_idx_e; sentence_position++) {
    int word = sen[sentence_position];
    if (word == -1) continue;
    float neu1 = 0;
    float neu1e = 0;
    next_random = next_random * (unsigned long)2514903917 + 11;
    int b = next_random % window;

    int cw = 0;
    for (int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      int c = sentence_position - window + a;
      if (c <  sent_idx_s) continue;
      if (c >= sent_idx_e) continue;
      int last_word = sen[c];
      if (last_word == -1) continue;
      neu1 += syn0[last_word * layer1_size + threadIdx.x];
      cw++;
    }

    if (cw) {
      neu1 /= cw;

      // HIERARCHICAL SOFTMAX
      if (hs) for (int d = vocab_codelen[word]; d < vocab_codelen[word+1]; d++) {
        int l2 = vocab_point[d] * layer1_size;

        if (threadIdx.x <  FSIZE) f[threadIdx.x] = neu1 * syn1[threadIdx.x + l2];
        __syncthreads();
        if (threadIdx.x >= FSIZE) f[threadIdx.x%(FSIZE)] += neu1 * syn1[threadIdx.x + l2];
        __syncthreads();
        for (int i=(FSIZE/2); i>0; i/=2) {
          if (threadIdx.x < i)
            f[threadIdx.x] += f[i + threadIdx.x];
          __syncthreads();
        }

        if      (f[0] <= -MAX_EXP) continue;
        else if (f[0] >=  MAX_EXP) continue;
        else if (threadIdx.x == 0) {
          f[0] = expTable[(int)((f[0] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (1 - vocab_code[d] - f[0]) * alpha;
        }
        __syncthreads();

        neu1e += g * syn1[threadIdx.x + l2];
        atomicAdd(&syn1[threadIdx.x + l2], g * neu1);
      }

      // NEGATIVE SAMPLING
      if (negative > 0) for (int d = 0; d < negative + 1; d++) {
        int target, label;
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target==0)    target = next_random % (vocab_size - 1) + 1;
          if (target==word) continue;
          label = 0;
        }
        int l2 = target * layer1_size;

        if (threadIdx.x <  FSIZE) f[threadIdx.x] = neu1 * syn1[threadIdx.x + l2];
        __syncthreads();
        if (threadIdx.x >= FSIZE) f[threadIdx.x%(FSIZE)] += neu1 * syn1[threadIdx.x + l2];
        __syncthreads();
        for (int i=(FSIZE/2); i>0; i/=2) {
          if (threadIdx.x < i)
            f[threadIdx.x] += f[i + threadIdx.x];
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          if (f[0] > MAX_EXP)
            g = (label - 1) * alpha;
          else if (f[0] < -MAX_EXP)
            g = (label - 0) * alpha;
          else
            g = (label - expTable[(int)((f[0]+MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        }
        __syncthreads();

        neu1e += g * syn1[l2 + threadIdx.x];
        atomicAdd(&syn1[l2 + threadIdx.x], g * neu1);
      }

      for (int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        int c = sentence_position - window + a;
        if (c <  sent_idx_s) continue;
        if (c >= sent_idx_e) continue;
        int last_word = sen[c];
        if (last_word == -1) continue;
        atomicAdd(&syn0[last_word * layer1_size + threadIdx.x], neu1e);
      }
    }
  }
}

void InitVocabStructCUDA()
{
  vocab_codelen = (int *)malloc((vocab_size + 1) * sizeof(int));
  vocab_codelen[0] = 0;
  for (int i = 1; i < vocab_size + 1; i++) 
    vocab_codelen[i] = vocab_codelen[i-1] + vocab[i-1].codelen;
  vocab_point = (int *)malloc(vocab_codelen[vocab_size] * sizeof(int));
  vocab_code = (char *)malloc(vocab_codelen[vocab_size] * sizeof(char));

  checkCUDAerr(cudaMalloc((void **)&d_vocab_codelen, (vocab_size + 1) * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_vocab_point, vocab_codelen[vocab_size] * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_vocab_code, vocab_codelen[vocab_size] * sizeof(char)));

  for (int i=0; i<vocab_size; i++) {
    for (int j=0; j<vocab[i].codelen; j++) {
      vocab_code[vocab_codelen[i] + j] = vocab[i].code[j];
      vocab_point[vocab_codelen[i] + j] = vocab[i].point[j];
    }   
  }   

  checkCUDAerr(cudaMemcpy(d_vocab_codelen, vocab_codelen, (vocab_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCUDAerr(cudaMemcpy(d_vocab_point, vocab_point, vocab_codelen[vocab_size] * sizeof(int), cudaMemcpyHostToDevice));
  checkCUDAerr(cudaMemcpy(d_vocab_code, vocab_code, vocab_codelen[vocab_size] * sizeof(char), cudaMemcpyHostToDevice));
}


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
  // FOR CUDA
  checkCUDAerr(cudaMalloc((void **)&d_table, table_size*sizeof(int)));
  checkCUDAerr(cudaMemcpy(d_table, table, table_size*sizeof(int), cudaMemcpyHostToDevice));
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) { if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  //  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[0], vocab_size, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words); 
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}
vector<vertex_id_t> id2offset;
void ReadVocabFromDegree(vector<vertex_id_t>& degrees){
  vertex_id_t v_num = degrees.size();
  long long a, i = 0;
  char word[MAX_STRING];
  for (a = 0; a < vocab_hash_size; a ++) vocab_hash[a] = -1;
  vocab_size = 0;
  for (vertex_id_t v = 0; v < v_num; v++)
  {
    std::sprintf(word,"%u",v);  // node ID 以字符串的形式存在 vocab 里面。
    a = AddWordToVocab(word);
    vocab[a].cn = degrees[v];
  }
  // 现在vocab 里面存了所有 {nodeId,degree} 的形式。
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  id2offset.resize(vocab_size);
  for(vertex_id_t vi = 0; vi<vocab_size; vi++){
    char* endptr;
    vertex_id_t nid = (vertex_id_t)strtoul(vocab[vi].word,&endptr,10);
    id2offset[nid] = vi;
  } 
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&last_emb, 128, (long long)vocab_size * layer1_size * sizeof(float));
  memset(last_emb,0,(long long)vocab_size * layer1_size * sizeof(float));

  if (last_emb == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
      syn1[a * layer1_size + b] = 0;
    checkCUDAerr(cudaMalloc((void **)&d_syn1, (long long)vocab_size * layer1_size * sizeof(float)));
    checkCUDAerr(cudaMemcpy(d_syn1, syn1, (long long)vocab_size * layer1_size * sizeof(float), cudaMemcpyHostToDevice));
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
      syn1neg[a * layer1_size + b] = 0;
    checkCUDAerr(cudaMalloc((void **)&d_syn1, (long long)vocab_size * layer1_size * sizeof(float)));
    checkCUDAerr(cudaMemcpy(d_syn1, syn1neg, (long long)vocab_size * layer1_size * sizeof(float), cudaMemcpyHostToDevice));
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
  }
  checkCUDAerr(cudaMalloc((void **)&d_syn0, (long long)vocab_size * layer1_size * sizeof(float)));
  checkCUDAerr(cudaMemcpy(d_syn0, syn0, (long long)vocab_size * layer1_size * sizeof(float), cudaMemcpyHostToDevice));

  CreateBinaryTree();
}

void cbowKernel(int *d_sen, int *d_sent_len, float alpha, int cnt_sentence, int reduSize)
{
  int bDim = layer1_size;
  int gDim = cnt_sentence;
  switch(reduSize) {
    case 128: cbow_kernel<64><<<gDim, bDim>>>
              (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
               d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
               d_sen, d_sent_len, d_syn1, d_syn0);
              break;
    case 256: cbow_kernel<128><<<gDim, bDim>>>
              (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
               d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
               d_sen, d_sent_len, d_syn1, d_syn0);
              break;
    case 512: cbow_kernel<256><<<gDim, bDim>>>
              (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
               d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
               d_sen, d_sent_len, d_syn1, d_syn0);
              break;
    default: printf("Can't support on vector size = %lld\n", layer1_size);
             exit(1);
             break;
  }

}

void sgKernel(int *d_sen, int *d_sent_len, int *d_negSample, float alpha, int cnt_sentence, int reduSize)
{
  int bDim= layer1_size;
  int gDim= cnt_sentence;

  if (reuseNeg) { // A sentence share negative samples
    dim3 bDimNeg(32, negative+1, 1);
    switch(layer1_size) {
      case 200: __sgNegReuse<200><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      case 300: __sgNegReuse<300><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      default: printf("Can't support on vector size = %lld\n", layer1_size);
               exit(1);
               break;
    }
  } else {
    switch(reduSize) {
      case 128: skip_gram_kernel<64><<<gDim, bDim>>>
                (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
                 d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
                 d_sen, d_sent_len, d_syn1, d_syn0);
                break;
      case 256: skip_gram_kernel<128><<<gDim, bDim>>>
                (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
                 d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
                 d_sen, d_sent_len, d_syn1, d_syn0);
                break;
      case 512: skip_gram_kernel<256><<<gDim, bDim>>>
                (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
                 d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
                 d_sen, d_sent_len, d_syn1, d_syn0);
                break;
      default: printf("Can't support on vector size = %lld\n", layer1_size);
               exit(1);
               break;
    }
  }
}

LR *lr_scheduler;
void TrainModelThread(string data_path)
{
  printf("[ p%d ]=====================Train file %s============\n",my_rank,data_path.c_str());
  long long word, word_count = 0, last_word_count = 0;
  long long local_iter = iter;

  // use in kernel
  int total_sent_len, reduSize= 32;
  int *sen, *sentence_length, *d_sen, *d_sent_len;
  sen = (int *)malloc(MAX_SENTENCE * 100 * sizeof(int));
  sentence_length = (int *)malloc((MAX_SENTENCE + 1) * sizeof(int));

  checkCUDAerr(cudaMalloc((void **)&d_sen, MAX_SENTENCE * 100 * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_sent_len, (MAX_SENTENCE + 1) * sizeof(int)));

  int *negSample = (int *)malloc(MAX_SENTENCE * negative * sizeof(int));
  int *d_negSample;
  checkCUDAerr(cudaMalloc(&d_negSample, MAX_SENTENCE * negative * sizeof(int)));

  while (reduSize < layer1_size) {
    reduSize *= 2;
  }

  clock_t now;
  strcpy(train_file,data_path.c_str());
  FILE *fi = fopen(train_file, "r");
  if(fi == nullptr) {
    printf("open [%s] fail\n",data_path.c_str());
    throw std::runtime_error("Data file open fail");
  }
  fseek(fi, 0, SEEK_SET);

  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  ", 13, alpha,
            word_count_actual / (float)(iter * train_words + 1) * 100,
            word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      // alpha = starting_alpha * (1 - word_count / (float)(iter * train_words + 1));
      // if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    total_sent_len = 0;
    sentence_length[0] = 0;
    int cnt_sentence = 0;

    while (cnt_sentence < MAX_SENTENCE) {                           // Read words
      int temp_sent_len = 0;
      char tSentence[MAX_SENTENCE_LENGTH];
      char *wordTok;
      if (feof(fi)) break;
      fgets(tSentence, MAX_SENTENCE_LENGTH + 1, fi);
      wordTok = strtok(tSentence, " \n\r\t");
      while(1) {
        if (wordTok == NULL) {
          word_count++;
          break;
        }
        word = SearchVocab(wordTok);
        wordTok = strtok(NULL, " \n\r\t");
        if (word == -1) continue;
        word_count++;
        if (word == 0) {
          word_count++;
          break;
        }
        if (sample > 0) {
          float ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          int next_random_t = rand();
          if (ran < (next_random_t & 0xFFFF) / (float)65536) continue;
        }
        sen[total_sent_len] = word;
        total_sent_len++;
        temp_sent_len++;
        if (temp_sent_len >= MAX_SENTENCE_LENGTH) break;
      }
      if (word == 0) {
        word_count++;
        break;
      }
      if (temp_sent_len >= MAX_SENTENCE_LENGTH) break;

      cnt_sentence++;
      sentence_length[cnt_sentence] = total_sent_len;
      if (total_sent_len >= (MAX_SENTENCE - 1) * 20) break;
    }

    if (feof(fi) || (word_count > train_words)) {                   // Initialize for iteration
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      for (int i=0; i<MAX_SENTENCE+1; i++)
        sentence_length[i] = 0;
      total_sent_len = 0;
      fseek(fi, 0, SEEK_SET);
      continue;
    }

    // Negative sampling in advance. A sentence shares negative samples
    for (int i=0; i<cnt_sentence * negative; i++) {
      int randd = rand();
      int tempSample = table[randd % table_size];
      if (tempSample == 0) negSample[i] = randd % (vocab_size - 1) + 1;
      else                 negSample[i] = tempSample;
    }
    checkCUDAerr(cudaMemcpy(d_negSample, negSample, cnt_sentence * negative * sizeof(int), cudaMemcpyHostToDevice));
    cudaError_t cet = cudaMemcpy(d_sen, sen, total_sent_len * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaSuccess != cet)
    {
      printf("%s %d : %s\n", __FILE__, __LINE__, cudaGetErrorString(cet));
      printf("copy size: %zu \n",total_sent_len*sizeof(int));
      exit(0);
    }
    checkCUDAerr(cudaMemcpy(d_sent_len, sentence_length, (cnt_sentence + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (cbow)
      cbowKernel(d_sen, d_sent_len, alpha, cnt_sentence, reduSize);
    else
      sgKernel(d_sen, d_sent_len, d_negSample, alpha, cnt_sentence, reduSize);
  }
  cudaDeviceSynchronize();
  checkCUDAerr(cudaMemcpy(syn0, d_syn0, vocab_size * layer1_size * sizeof(float), cudaMemcpyDeviceToHost));

  fclose(fi);

  // free memory
  free(sen);
  free(sentence_length);
  free(negSample);
  cudaFree(d_sen);
  cudaFree(d_sent_len);
  cudaFree(d_negSample);
}
vector<vertex_id_t> g_v_degree;

void myIntersectition(const vector<vertex_id_t>& v1,const vector<vertex_id_t>& v2,vector<vertex_id_t>& v_intersection)
{
    int p1=0;
    int p2=0;
    int v1_sz = v1.size();
    int v2_sz = v2.size();
    const vector<vertex_id_t>*long_v;
    const vector<vertex_id_t>*short_v;
    if(v1_sz>v2_sz){
        long_v=&v1;
        short_v=&v2;
    }else{
        long_v=&v2;
        short_v=&v1;
    }
    int max_sz = max(v1_sz,v2_sz);
    int min_sz = min(v1_sz,v2_sz);
    
    int begin = 0;
    if(v1.empty()||v2.empty())return;
    if((*long_v)[max_sz-1]<(*short_v)[0]||((*long_v)[0]>(*short_v)[min_sz-1]))return;
    while(p1<max_sz&&p2<min_sz){
        int offset = 1;
        int last_p = offset;
        if((*short_v)[p2]<((*long_v)[p1])){
            p2++;
            continue;
        }
        while((*long_v)[p1+offset-1]<(*short_v)[p2]){
            offset=offset*2;
            last_p = p1+offset<max_sz?offset:max_sz-p1;
            if(p1+offset>=max_sz)break;
        }
        if((*long_v)[max_sz-1]<(*short_v)[p2]){
            p2++;
            break;
        }
        auto iter = lower_bound(long_v->begin()+(p1+offset/2),long_v->begin()+p1+last_p,(*short_v)[p2]); // 如果在 区间找到了
        int t = iter - long_v->begin();
        if(*iter==(*short_v)[p2]){
            v_intersection.push_back((*short_v)[p2]);
            p2++;
            p1=t++;
        }else{
            p2++;
            p1=(p1+offset/2);
        }
    };   
   
}

float cos_sim(float* v1,float* v2, int dim){
  float dot_product = 0.0;
  float v1_l2 = 0.0, v2_l2 = 0.0;
  for(int d = 0; d < dim; d++){
    dot_product += v1[d] * v2[d];
    v1_l2 += v1[d] * v1[d];
    v2_l2 += v2[d] * v2[d];
  }
  v1_l2 = sqrt(v1_l2);
  v2_l2 = sqrt(v2_l2);
  return dot_product/(v1_l2 * v2_l2);
}
float node_neighbour_average_cos_sim(vertex_id_t v_id,myEdgeContainer*csr){
  float sum_cos_sim = 0.0f;
  int nei_n = csr->adj_lists[v_id].end - csr->adj_lists[v_id].begin;
  // 1. get neighbour set;
  for(auto it = csr->adj_lists[v_id].begin; it < csr->adj_lists[v_id].end; it++){
    vertex_id_t nei = it->neighbour;
    vertex_id_t v_1 = id2offset[v_id];
    vertex_id_t v_2 = id2offset[nei];
    float sim = cos_sim(syn0+v_1 * layer1_size, syn0+v_2*layer1_size,layer1_size);
    sum_cos_sim += sim;
  }
  return sum_cos_sim / nei_n;
}

float find_supernode_topK_accurancy(float p,int k,myEdgeContainer*csr){
  float top_sum = 0;
  for(vertex_id_t v_i = 0; v_i < vocab_size*0.03; v_i ++){
    vector<std::pair<float,vertex_id_t>> supernode_sim;
    for(vertex_id_t v_j = 0; v_j < vocab_size * p; v_j ++){
      float sim = cos_sim(syn0+v_i * layer1_size, syn0+v_j*layer1_size,layer1_size);
      supernode_sim.push_back({sim,v_j});
    }
    sort(supernode_sim.begin(),supernode_sim.end(),[](std::pair<float,vertex_id_t>&p1,std::pair<float,vertex_id_t>&p2){
        return p1.first > p2.first;
        });
    vector<vertex_id_t> selected_topK(k);
    for(int i = 0 ;i< k; i++){
      // selected_topK[i] = supernode_sim[i].second;
      selected_topK[i] = supernode_sim[i].second ; 
    }
    vector<vertex_id_t> real_neighbor;
    for(auto it = csr->adj_lists[v_i].begin; it < csr->adj_lists[v_i].end; it++){
      real_neighbor.push_back(it->neighbour);
    }
    sort(real_neighbor.begin(),real_neighbor.end());
    sort(selected_topK.begin(),selected_topK.end());
    vector<vertex_id_t>result_set;
    myIntersectition(selected_topK, real_neighbor, result_set);
    top_sum += result_set.size();
  }
  return top_sum/ (float)(k * vocab_size *p);
}
void TrainModel(SyncQueue& taskq,myEdgeContainer*csr) {
  printf("==========================Train Model In=====================\n");
  long a, b, c, d;
  FILE *fo;
  starting_alpha = alpha;
  ReadVocabFromDegree(g_v_degree);
  printf("========================Read Vocab ok=======================\n");
  printf("vocab_size: %lu\n",vocab_size);

  // for(size_t i = 0; i < vocab_size * 0.10;i++){
  //   printf("id: %s, degree: %ld\n",vocab[i].word,vocab[i].cn);
  // }

  vector<float> H;
  float delta_H;
  

  // if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  // if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) printf("[ Warning ] output file missing\n");
  if (output_file[0] == 0) return;
  InitNet();
  printf("[ %d ] InitNet Success\n",my_rank);
  if (hs > 0) InitVocabStructCUDA();
  if (negative > 0) InitUnigramTable();

  start = clock();
  srand(time(NULL));

  printf("==========init success================\n");
  float* kl = new float;
  // lr_scheduler = new  FixedLR(0.025);
  // lr_scheduler = new  StepDecayLR(0.025,0.5,3);
  lr_scheduler = new ExponentialDecayLR(0.025,0.1);

  // TrainModelThread("./out/tmp-0-1.txt");
  // float acc = find_supernode_topK_accurancy(0.01,10,csr);
  //
  // cout << "find super node topK acc: " << acc << endl;

  FILE* f_nei_cos_sim = fopen("neighbour_average_cos_sim.txt","w");
  vector<vector<float>>node_neighbour_average_cos_sim_array(vocab_size);
  int n2 = 2;
  while(n2++ < 30){
    char fc[100];
    sprintf(fc,"./out/tmp-0-%d.txt",n2);
    alpha = lr_scheduler->get_lr();
    TrainModelThread(fc);
    std::cout << std::endl;
    for(vertex_id_t v = 0;v < vocab_size; v++){
      float s = node_neighbour_average_cos_sim(v,csr);
      node_neighbour_average_cos_sim_array[v].push_back(s);
    }
  }
  for(vertex_id_t i = 0; i < vocab_size; i++){
    for(int j = 0; j < node_neighbour_average_cos_sim_array[i].size();j++){
      fprintf(f_nei_cos_sim, "%.3f ",node_neighbour_average_cos_sim_array[i][j]);
    }
    fprintf(f_nei_cos_sim,"\n");
  }
  fclose(f_nei_cos_sim);
  float p = 0.6;
  vector<bool> flag(vocab_size,true);
  int sum_count = 0;
  for(int s = 0;s < node_neighbour_average_cos_sim_array[0].size();s++){
    int count = 0;
    for(vertex_id_t v = 0;v < vocab_size;v++){
      if(node_neighbour_average_cos_sim_array[s][v] > p && flag[v] == true){
        count++;
        flag[v] = false;
      }
    }
    cout <<count <<" ";
    sum_count+=count;
  }
  cout << " sum: "<< sum_count << endl;
  return;

  FILE* f_topk = fopen("super_topK.txt","w");
  int n1 = 2;
  while(n1++<30){
    char fc[100];
    sprintf(fc,"./out/tmp-0-%d.txt",n1);
    alpha = lr_scheduler->get_lr();
    TrainModelThread(fc);
    std::cout << std::endl;
    float acc = find_supernode_topK_accurancy(0.01,10,csr);
    cout << "find super node topK acc: " << acc << endl;
    fprintf(f_topk,"%f\n",acc);
  }
  fclose(f_topk);
  return;

  // [Test]
  
  //compute_kl_from_emb(last_emb,syn0, kl,tmpN,layer1_size);
  //std::cout << std::endl;
  //checkCUDAerr(cudaDeviceSynchronize()); 
  //printf("[ %d ] COMPUTE KL From Emb : %.f\n",my_rank,*kl);

  // [End Test]

 //  FILE* f_rel_ent = fopen("rel_ent.txt","w");
 //  FILE* f_delta_ent = fopen("delta_ent.txt","w");
 // int n = 2;
 //  while(n++<30){
 //    char fc[100];
 //    sprintf(fc,"./out/tmp-0-%d.txt",n);
 //    TrainModelThread(fc);
 //    cout << endl;
 //    // termination judgment
 //    compute_kl_from_emb(last_emb,syn0, kl,vocab_size * 0.1,layer1_size);
 //    checkCUDAerr(cudaDeviceSynchronize()); 
 //    printf("[ %d ] COMPUTE KL from emb : %.f\n",my_rank,*kl);
 //    fprintf(f_rel_ent,"%s: %f\n",fc,*kl);
 //    memcpy(last_emb,syn0,(long long)vocab_size * layer1_size * sizeof(float));
 //  }
 //  fclose(f_delta_ent);
 //  fclose(f_rel_ent);
 //  printf("=============Task over | Calculate delta H ===========\n");
  
  
  cudaFree(d_table);
  cudaFree(d_syn1);
  cudaFree(d_syn0);
  cudaFree(d_vocab_codelen);
  cudaFree(d_vocab_point);
  cudaFree(d_vocab_code);

  if(my_rank != 0) return;

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  fo = fopen(output_file, "wb");
  if(fo == NULL) printf("[ %d ] [%s] open fail\n",output_file);
  if (classes == 0) {	
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    float closev, x;
    float *cent = (float *)calloc(classes * layer1_size, sizeof(float));

    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }

    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);


    free(centcn);
    free(cent);
    free(cl);
  }
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout<<"[ "<<my_rank<<" ] Save Embedding: " <<time_span.count() << " s" <<std::endl;
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}
int train_corpus_cuda(int argc, char **argv,const vector<vertex_id_t>& degrees,SyncQueue& corpus_q,int _my_rank,myEdgeContainer* csr) {

  printf("No.10%: %llu\n",degrees[degrees.size() * 0.03]);
  // test area 

  // test __global__ void cosine_similarity_kernel(float *d_vectors, float *d_result, int v, int dim){
  float h_vec[] = {1,2,3,1,4,1};
  float h_vec2[] = {1,1,1,1,2,1};
  float *d_vec;
  float *d_result;
  int v = 2;
  int dim = 3;
  float* h_result = new float[v * dim];

  checkCUDAerr(cudaMalloc((void **)&d_result, (long long)v * dim * sizeof(float)));
  checkCUDAerr(cudaMalloc((void **)&d_vec, (long long)v * dim * sizeof(float)));
  checkCUDAerr(cudaMemcpy(d_vec, h_vec, (long long)v * dim * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockSize(16,16);
	dim3 gridSize((v+blockSize.x-1) / blockSize.x,(v+blockSize.y-1)/blockSize.y);
  cosine_similarity_kernel<<<gridSize,blockSize>>>(d_vec,d_result,v,dim);

  checkCUDAerr(cudaMemcpy(h_result, d_result, (long long)v * dim * sizeof(float), cudaMemcpyDeviceToHost));

  for(int i = 0; i < v * v; i++){
    std::cout << h_result[i] << " ";
  }
  std::cout << std::endl;

  checkCUDAerr(cudaMemcpy(d_vec, h_vec2, (long long)v * dim * sizeof(float), cudaMemcpyHostToDevice));
  cosine_similarity_kernel<<<gridSize,blockSize>>>(d_vec,d_result,v,dim);
  checkCUDAerr(cudaMemcpy(h_result, d_result, (long long)v * dim * sizeof(float), cudaMemcpyDeviceToHost));
  // TEST __global__ void compute_kl_divergence_kernel(float *d_A,float *d_B, float *d_result, int v){
  for(int i = 0; i < v * v; i++){
    std::cout << h_result[i] << " ";
  }
  std::cout << std::endl;
  
  float h_A[] = {1,0.942809,0.942809,1 };
  float h_B[] = {1,0.755929,0.755929,1 };
  float *d_A,*d_B,*d_res;
  float *h_res = new float;
  
  checkCUDAerr(cudaMalloc((void **)&d_A, (long long)v * v * sizeof(float)));
  checkCUDAerr(cudaMalloc((void **)&d_B, (long long)v * v * sizeof(float)));
  checkCUDAerr(cudaMemcpy(d_A, h_A, (long long)v * v * sizeof(float), cudaMemcpyHostToDevice));
  checkCUDAerr(cudaMemcpy(d_B, h_B, (long long)v * v * sizeof(float), cudaMemcpyHostToDevice));
  checkCUDAerr(cudaMalloc((void **)&d_res,sizeof(float)));

  int blockSize2 = 256;
  int gridSize2 = (v * v + blockSize2 - 1) / blockSize2;
  compute_kl_divergence_kernel<<<gridSize2,blockSize2>>>(d_A,d_B,d_res,v);
  checkCUDAerr(cudaMemcpy(h_res,d_res,sizeof(float),cudaMemcpyDeviceToHost));

  std::cout<< "KL: " << *h_res<< std::endl;
  

  // TEST void  compute_kl_from_emb(float *emb1, float *emb2,float* h_result, int v,int dim){
  float h_test1[] = {1,1,1,1,2,1};
  float h_test2[] = {1,2,3,1,4,1};

  float* h_kl = new float;
  compute_kl_from_emb(h_test1,h_test2,h_kl,2,3);
  printf("[TEST] GET Kl From Emb: %f\n",*h_kl);

  // end test area

  my_rank = _my_rank;
  g_v_degree.assign(degrees.begin(), degrees.end());
  printf("train_corpus_Cuda calling!!!!\n");
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-reuse-neg <int>\n");
    printf("\t\tA sentence share a negative sample set; (0 = not used / 1 = used)\n");

    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }

  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-emb_output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-reuse-neg", argc, argv)) > 0) reuseNeg = atoi(argv[i + 1]);

  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));

  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  checkCUDAerr(cudaMalloc((void **)&d_expTable, (EXP_TABLE_SIZE + 1) * sizeof(float)));
  checkCUDAerr(cudaMemcpy(d_expTable, expTable, (EXP_TABLE_SIZE + 1) * sizeof(float), cudaMemcpyHostToDevice));

  TrainModel(corpus_q,csr);

  // memory free
  free(vocab_codelen);
  free(vocab_point);
  free(vocab_code);
  free(table);
  free(syn0);
  free(syn1);
  free(syn1neg);
  free(vocab);
  free(vocab_hash);
  free(expTable);
  cudaFree(d_expTable);
  free(last_emb);

  return 0;
}
