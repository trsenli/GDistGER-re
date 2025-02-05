 
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <type_traits>
#include <thread>
#include <random>
#include <map>
#include <functional>
#include <unistd.h>
#include <sys/mman.h>
#include <limits.h>
// #include <boost/functional/hash.hpp>

#include <omp.h>

#include "type.hpp"
#include "util.hpp"
#include "constants.hpp"
#include "storage.hpp"
#include "mpi_helper.hpp"
using namespace std;

// #define PERF_PROF true
#define V_alpha 5

template<typename edge_data_t>
struct AdjUnit
{
    vertex_id_t neighbour;
    edge_data_t data;
};

template<>
struct AdjUnit<EmptyData>
{
    union
    {
        vertex_id_t neighbour;
        EmptyData data;
    };
};


template<typename edge_data_t>
struct AdjList
{

    AdjUnit<edge_data_t> *begin;
    AdjUnit<edge_data_t> *end;
    void init()
    {
        begin = nullptr;
        end = nullptr;
    }
    void printAdjList()
    {
        for(AdjUnit<edge_data_t> *p = begin; p < end; p++)
        {
            std::cout << p->neighbour << " " ;
        }
        std::cout << std::endl;
    }
};

//comprised column row
template<typename edge_data_t>
struct EdgeContainer
{

    AdjList<edge_data_t> *adj_lists;

    AdjUnit<edge_data_t> *adj_units;
    EdgeContainer() : adj_lists(nullptr), adj_units(nullptr) {}
    ~EdgeContainer()
    {
        if (adj_lists != nullptr)
        {
            delete []adj_lists;
        }
        if (adj_units != nullptr)
        {
            delete []adj_units;
        }
    }
};

struct COAdjList
{
    edge_id_t begin;
    edge_id_t end;
    
};

class CoOccorCsr
{
public:
    vector<COAdjList> adjList;
    vector<vertex_id_t>adjUnit;
    vector<edge_id_t> co_occor;
    void add_co_occor(vertex_id_t src,vertex_id_t dst)
    {
        edge_id_t begin = adjList[src].begin;
        edge_id_t end = adjList[src].end;
        auto co_occor_vertex = find(adjUnit.begin()+begin,adjUnit.begin()+end,dst);
        if(co_occor_vertex>= adjUnit.begin()+end || co_occor_vertex < adjUnit.begin()+begin){
            printf("[p %d error] src: %u  dst: %u \n",get_mpi_rank(),src,dst);
        }
        assert(co_occor_vertex < adjUnit.begin()+end);
	    assert(co_occor_vertex >= adjUnit.begin() + begin);
        edge_id_t idx = co_occor_vertex - adjUnit.begin();
        co_occor[idx]++;
    }
};

enum MPIMessageTag {
    Tag_ShuffleGraph,
    Tag_Msg,
    Tag_Msg_Count
};

template<typename T>
class Message
{
public:
    vertex_id_t dst_vertex_id;
    T data;
};

struct DistributedExecutionCtx
{
    std::mutex phase_locks[DISTRIBUTEDEXECUTIONCTX_PHASENUM];
    int unlocked_phase;
    size_t **progress;
public:
    DistributedExecutionCtx()
    {
        progress = nullptr;
    }
};

enum GraphFormat
{
    GF_Binary,
    GF_Edgelist
};

template<typename edge_data_t>
class GraphEngine
{
// protected: 
public:

    vertex_id_t v_num;

    edge_id_t e_num;
    int worker_num;

    edge_id_t local_e_num;

    vertex_id_t *vertex_partition_begin;
    vertex_id_t *vertex_partition_end;

    partition_id_t local_partition_id;

    partition_id_t partition_num;


    // map<pair<vertex_id_t, vertex_id_t>, int> commonNeighbors; 
    unordered_map<com_neighbour_t, int> commonNeighbors;
    MessageBuffer **thread_local_msg_buffer; 
    MessageBuffer **msg_send_buffer;
    MessageBuffer **msg_recv_buffer;
    std::mutex *send_locks;
    std::mutex *recv_locks;

    DistributedExecutionCtx dist_exec_ctx;
public:

    vertex_id_t *vertex_in_degree;

    vertex_id_t *vertex_out_degree;

    vertex_id_t *vertex_freq;

    partition_id_t *vertex_partition_id;
    // 
    EdgeContainer<edge_data_t> *csr;
    // global graph csr
    EdgeContainer<edge_data_t> *g_csr;

    CoOccorCsr *co_occor;

protected:

    void set_graph_engine_concurrency(int worker_num_param)
    {
        this->worker_num = worker_num_param;
        omp_set_dynamic(0);
        omp_set_num_threads(worker_num);
        //message buffer depends on worker number
        free_msg_buffer();
    }

public:

    inline bool is_local_vertex(vertex_id_t v_id)
    {
        return v_id >= vertex_partition_begin[local_partition_id]
            && v_id < vertex_partition_end[local_partition_id];
    }

    inline bool is_valid_edge(Edge<edge_data_t> e)
    {
        return e.src < v_num && e.dst < v_num;
    }

    inline vertex_id_t get_vertex_num()
    {
        return v_num;
    }

    inline edge_id_t get_edge_num()
    {
        return e_num;
    }

    inline int get_worker_num()
    {
        return worker_num;
    }

    inline vertex_id_t get_local_vertex_begin()
    {
        return vertex_partition_begin[local_partition_id];
    }

    inline vertex_id_t get_local_vertex_end()
    {
        return vertex_partition_end[local_partition_id];
    }

    inline vertex_id_t get_vertex_begin(partition_id_t p)
    {
        return vertex_partition_begin[p];
    }

    inline vertex_id_t get_vertex_end(partition_id_t p)
    {
        return vertex_partition_end[p];
    }

public:

    template<typename T>
    void dealloc_vertex_array(T * array)
    {
        dealloc_array(array, v_num);
    }

    template<typename T>
    void dealloc_array(T * array, size_t num)
    {
        munmap(array, sizeof(T) * num);
    }


    template<typename T>
    T * alloc_vertex_array()
    {
        return alloc_array<T>(v_num);
    }

    template<typename T>
    T * alloc_array(size_t num)
    {
        T* array = (T*) mmap(NULL, sizeof(T) * num, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(array != nullptr);
        return array;
    }

    GraphEngine()
    {
        vertex_partition_begin = nullptr;
        vertex_partition_end = nullptr;

        thread_local_msg_buffer = nullptr;
        msg_send_buffer = nullptr;
        msg_recv_buffer = nullptr;

        send_locks = nullptr;
        recv_locks = nullptr;

        vertex_in_degree = nullptr;
        vertex_out_degree = nullptr;
        vertex_freq = nullptr;

        vertex_partition_id = nullptr;

        csr = nullptr;
 
        this->worker_num = std::max(1, ((int)std::thread::hardware_concurrency()) - 1);

        omp_set_dynamic(0);
        omp_set_num_threads(worker_num);
    }

    virtual ~GraphEngine()
    {
        if (vertex_partition_begin != nullptr)
        {
            delete []vertex_partition_begin;
        }
        if (vertex_partition_end != nullptr)
        {
            delete []vertex_partition_end;
        }

        if (send_locks != nullptr)
        {
            delete []send_locks;
        }
        if (recv_locks != nullptr)
        {
            delete []recv_locks;
        }

        if (vertex_in_degree != nullptr)
        {
            dealloc_vertex_array<vertex_id_t>(vertex_in_degree);
        }
        if (vertex_out_degree != nullptr)
        {
            dealloc_vertex_array<vertex_id_t>(vertex_out_degree);
        }
        if(vertex_freq != nullptr)
        {
            dealloc_vertex_array<vertex_id_t>(vertex_freq);
        }
        if (vertex_partition_id != nullptr)
        {
            dealloc_vertex_array<partition_id_t>(vertex_partition_id);
        }

        if (csr != nullptr)
        {
            delete csr;
        }
        if(co_occor != nullptr)
            delete co_occor;
        if (dist_exec_ctx.progress != nullptr)
        {
            for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
            {
                delete []dist_exec_ctx.progress[t_i];
            }
            delete []dist_exec_ctx.progress;
        }

        free_msg_buffer();
    }

    void printEdgeContainer(AdjList<edge_data_t> *adj_lists)
    {
        for(int i = 0; i < this->v_num; i++ )
        {
            adj_lists[i].printAdjList();
        }
    }

    void g_build_edge_container(Edge<edge_data_t>* edges,edge_id_t local_edge_num, vertex_id_t* vertex_out_degree)
    {
        this->co_occor->adjList.resize(v_num);
        this->co_occor->adjUnit.resize(e_num + v_num);
        this->co_occor->co_occor.resize(e_num + v_num);

        for(int i =0; i<e_num + v_num;i++){
            this->co_occor->adjUnit[i] = 0;
            this->co_occor->co_occor[i] = 0;
        }

        vector<edge_id_t> edge_partion_begin(partition_num,0);
        vector<edge_id_t> edge_partion_end(partition_num,0);

        edge_id_t edge_chunk=local_edge_num + vertex_partition_end[local_partition_id] - vertex_partition_begin[local_partition_id];
        printf("edge chunk %d %zu\n",local_partition_id,edge_chunk);
        
        vector<edge_id_t> edge_chunk_partition(partition_num,0);
        edge_chunk_partition[local_partition_id] = edge_chunk;
        MPI_Allreduce(MPI_IN_PLACE,edge_chunk_partition.data(),partition_num,get_mpi_data_type<edge_id_t>(),MPI_SUM,MPI_COMM_WORLD);
        printf("%d | %zu %zu %zu %zu\n",local_partition_id,edge_chunk_partition[0],edge_chunk_partition[1],edge_chunk_partition[2],edge_chunk_partition[3]);
        edge_id_t e_offset=0;
        for(int p = 0;p < partition_num ;p ++)
        {
            edge_partion_begin[p] =e_offset ;
            
            edge_partion_end[p] = edge_partion_begin[p] + edge_chunk_partition[p];

            e_offset += edge_chunk_partition[p];

        }

        edge_id_t edge_offset = edge_partion_begin[local_partition_id];
        printf("%d %zu \n",local_partition_id,edge_offset);
        edge_id_t chunk_edge_idx = 0;
         for (vertex_id_t v_i = vertex_partition_begin[local_partition_id]; v_i < vertex_partition_end[local_partition_id]; v_i++)
        {
            this->co_occor->adjList[v_i].begin = chunk_edge_idx + edge_offset;
            this->co_occor->adjUnit[ co_occor->adjList[v_i].begin ] = v_i;
            this->co_occor->adjList[v_i].end = this->co_occor->adjList[v_i].begin + 1;
            chunk_edge_idx += vertex_out_degree[v_i];
            chunk_edge_idx += 1;
        }



        for (edge_id_t e_i = 0; e_i < local_edge_num; e_i++)
        {
            auto e = edges[e_i];
            // auto ep = ec->adj_lists[e.src].end++; 
            // ep->neighbour = e.dst;
            // ep->data = e.data;

            auto& adjL = this->co_occor->adjList[e.src];
            auto& adjU = this->co_occor ->adjUnit;
            // auto p = find(adjU.begin() + adjL.begin,adjU.begin() + adjL.end,e.dst);
            // if(p != adjU.begin() + adjL.end)
            //     continue;

            auto t = this->co_occor->adjList[e.src].end;
            this->co_occor->adjUnit[t] = e.dst;
            this->co_occor->co_occor[t] = 0;
            this->co_occor->adjList[e.src].end += 1 ;
        }
        printf("%d g_build_local\n",local_partition_id);

        // if(local_partition_id == 0)
        // {
        //      printf("[0]before %lu %lu \n",this->co_occor->adjList[0].begin,this->co_occor->adjList[0].end);
        //     printf("[0]before %lu %lu \n",this->co_occor->adjList[v_num -1].begin,this->co_occor->adjList[v_num -1].end);
        // }  
        //  if(local_partition_id == 3)
        // {
        //      printf("[3]before %lu %lu \n",this->co_occor->adjList[0].begin,this->co_occor->adjList[0].end);
        //     printf("[3]before %lu %lu \n",this->co_occor->adjList[v_num -1].begin,this->co_occor->adjList[v_num -1].end);
        // }    
        MPI_Allreduce(MPI_IN_PLACE,(edge_id_t*)this->co_occor->adjList.data(),this->co_occor->adjList.size()*2,get_mpi_data_type<edge_id_t>(),MPI_SUM,MPI_COMM_WORLD);
        // if(local_partition_id == 0)
        // {
        //     printf("[0]after %lu %lu \n",this->co_occor->adjList[0].begin,this->co_occor->adjList[0].end);
        //     printf("[0]after %lu %lu \n",this->co_occor->adjList[v_num-1].begin,this->co_occor->adjList[v_num-1].end);
        // }
        // if(local_partition_id == 3)
        // {
        //     printf("[3]after %lu %lu \n",this->co_occor->adjList[0].begin,this->co_occor->adjList[0].end);
        //     printf("[3]after %lu %lu \n",this->co_occor->adjList[v_num-1].begin,this->co_occor->adjList[v_num-1].end);
        // }

        int round_size = 100000000;
        int round = this->co_occor->adjUnit.size()/round_size;
        int round_mod = this->co_occor->adjUnit.size()%round_size;
        for(int i =0;i < round; i++){
            MPI_Allreduce(MPI_IN_PLACE,this->co_occor->adjUnit.data()+i*round_size,round_size,get_mpi_data_type<vertex_id_t>(),MPI_MAX,MPI_COMM_WORLD);
        }
        MPI_Allreduce(MPI_IN_PLACE,this->co_occor->adjUnit.data()+round*round_size,round_mod,get_mpi_data_type<vertex_id_t>(),MPI_MAX,MPI_COMM_WORLD);

    }
     void g_build_edge_container(Edge<edge_data_t>* edges, edge_id_t e_num, vertex_id_t v_num, EdgeContainer<edge_data_t>* ec, vertex_id_t* vertex_out_degree)
     {
         this->co_occor->adjList.resize(v_num);
         this->co_occor->adjUnit.resize(e_num);
         this->co_occor->co_occor.resize(e_num);

         ec->adj_lists = new AdjList<edge_data_t>[v_num];
         ec->adj_units = new AdjUnit<edge_data_t>[e_num];

         edge_id_t chunk_edge_idx = 0;
         for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
         {
             ec->adj_lists[v_i].begin = ec->adj_units + chunk_edge_idx;
             ec->adj_lists[v_i].end = ec->adj_lists[v_i].begin; 

             this->co_occor->adjList[v_i].begin = chunk_edge_idx;
             this->co_occor->adjList[v_i].end = this->co_occor->adjList[v_i].begin;

             chunk_edge_idx += vertex_out_degree[v_i];
         }
         printf("stage 1\n");
         for (edge_id_t e_i = 0; e_i < e_num; e_i++)
         {
             auto e = edges[e_i];

             auto ep = ec->adj_lists[e.src].end++; 
             ep->neighbour = e.dst;
             ep->data = e.data;

             auto t = this->co_occor->adjList[e.src].end++;
             this->co_occor->adjUnit[t] = e.dst;
             this->co_occor->co_occor[t] = 0;

         }
     }

    void build_edge_container(Edge<edge_data_t> *edges, edge_id_t local_edge_num, EdgeContainer<edge_data_t> *ec, vertex_id_t* vertex_out_degree)
    {

        ec->adj_lists = new AdjList<edge_data_t>[v_num];

        ec->adj_units = new AdjUnit<edge_data_t>[local_edge_num];
        edge_id_t chunk_edge_idx = 0;
        for (vertex_id_t v_i = vertex_partition_begin[local_partition_id]; v_i < vertex_partition_end[local_partition_id]; v_i++)
        {
            ec->adj_lists[v_i].begin = ec->adj_units + chunk_edge_idx;
            ec->adj_lists[v_i].end = ec->adj_lists[v_i].begin;

            chunk_edge_idx += vertex_out_degree[v_i];
        }

        for (edge_id_t e_i = 0; e_i < local_edge_num; e_i++)
        {
            auto e = edges[e_i];
            auto ep = ec->adj_lists[e.src].end ++;
           
            ep->neighbour = e.dst;
            
            if (!std::is_same<edge_data_t, EmptyData>::value)
            {
                ep->data = e.data;
                // std::cout << e.data << endl;
            }else{
                std::cout << " no common neighbor " << std::endl;
                exit(0);
            }
        }
    }
   
    void shuffle_edges(Edge<edge_data_t> *misc_edges, edge_id_t misc_e_num, Edge<edge_data_t> *local_edges, edge_id_t local_e_num)
    {

        std::vector<edge_id_t> e_count(partition_num, 0);   
        for (edge_id_t e_i = 0; e_i < misc_e_num; e_i++)
        {

            e_count[vertex_partition_id[misc_edges[e_i].src]]++;
        }

        Edge<edge_data_t> *tmp_es  = new Edge<edge_data_t>[misc_e_num];

        std::vector<edge_id_t> e_p(partition_num, 0);
        for (partition_id_t p_i = 1; p_i < partition_num; p_i++)
        {
            e_p[p_i] = e_p[p_i - 1] + e_count[p_i - 1];
        }

        auto e_begin = e_p;
        for (edge_id_t e_i = 0; e_i < misc_e_num; e_i++)
        {

            auto pt = vertex_partition_id[misc_edges[e_i].src];

            tmp_es[e_p[pt] ++] = misc_edges[e_i];
        }

        edge_id_t local_edge_p = 0;

        std::thread send_thread([&](){

            for (partition_id_t step = 0; step < partition_num; step++)
            {

                partition_id_t dst = (local_partition_id + step) % partition_num;

                size_t tot_send_sz = e_count[dst] * sizeof(Edge<edge_data_t>);
#ifdef UNIT_TEST
                const int max_single_send_sz = (1 << 8) / sizeof(Edge<edge_data_t>) * sizeof(Edge<edge_data_t>);
#else

                const int max_single_send_sz = (1 << 28) / sizeof(Edge<edge_data_t>) * sizeof(Edge<edge_data_t>);
                // if(this->local_partition_id == 0)
                // {
                //     std::cout << "max_single_send_sz = " << max_single_send_sz << std::endl;
                // }
#endif

                void* send_data = tmp_es + e_begin[dst];
                while (true)
                {
                    
                    MPI_Send(&tot_send_sz, 1, get_mpi_data_type<size_t>(), dst, Tag_ShuffleGraph, MPI_COMM_WORLD);
                    if (tot_send_sz == 0)
                    {
                        break;
                    }

                    int send_sz = std::min((size_t)max_single_send_sz, tot_send_sz);
                    tot_send_sz -= send_sz;
                    
                    MPI_Send(send_data, send_sz, get_mpi_data_type<char>(), dst, Tag_ShuffleGraph, MPI_COMM_WORLD);
                    send_data = (char*)send_data + send_sz;
                }
                usleep(100000);
            }
        });

        std::thread recv_thread([&](){
            for (partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t src = (local_partition_id + partition_num - step) % partition_num;
                while (true)
                {
                   
                    size_t remained_sz;
                    MPI_Recv(&remained_sz, 1, get_mpi_data_type<size_t>(), src, Tag_ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (remained_sz == 0)
                    {
                        break;
                    }
                    
                    MPI_Status recv_status;
                    MPI_Probe(src, Tag_ShuffleGraph, MPI_COMM_WORLD, &recv_status);
                    int sz;
                    MPI_Get_count(&recv_status, get_mpi_data_type<char>(), &sz);

                    MPI_Recv(local_edges + local_edge_p, sz, get_mpi_data_type<char>(), src, Tag_ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    local_edge_p += sz / sizeof(Edge<edge_data_t>);
                }
                usleep(100000);
            }
        });

        send_thread.join();
        recv_thread.join();
        delete []tmp_es;
        assert(local_e_num == local_edge_p);
    }

    void load_graph(vertex_id_t v_num_param, const char* graph_path,const char* partition_path, bool load_as_undirected = false, GraphFormat graph_format = GF_Binary)
    {
        Timer timer;

        this->v_num = v_num_param;
        this->partition_num = get_mpi_size();
        this->local_partition_id = get_mpi_rank();
        this->local_e_num = 0;

        Edge<edge_data_t> *read_edges;
        edge_id_t read_e_num;

        Edge<edge_data_t> *g_read_edges;
        edge_id_t g_read_e_num;

        if (graph_format == GF_Binary)
        {
            read_graph(graph_path, local_partition_id, partition_num, read_edges, read_e_num);
            //g_read_graph(graph_path,  g_read_edges, g_read_e_num);
            printf("read edge=========ok\n");
        } else if (graph_format == GF_Edgelist)
        {

            read_edgelist(graph_path, local_partition_id, partition_num, read_edges, read_e_num);
        } else
        {
            fprintf(stderr, "Unsupported graph formant");
            exit(1);
        }
        if (load_as_undirected)
        {
            Edge<edge_data_t> *undirected_edges = new Edge<edge_data_t>[read_e_num * 2];
#pragma omp parallel for
            for (edge_id_t e_i = 0; e_i < read_e_num; e_i++)
            {

                undirected_edges[e_i * 2] = read_edges[e_i];

                std::swap(read_edges[e_i].src, read_edges[e_i].dst);
                undirected_edges[e_i * 2 + 1] = read_edges[e_i];
            }
            delete []read_edges;
            read_edges = undirected_edges;
            read_e_num *= 2;
        }

        this->vertex_out_degree = alloc_vertex_array<vertex_id_t>();
        this->vertex_in_degree = alloc_vertex_array<vertex_id_t>();
        this->vertex_freq = alloc_vertex_array<vertex_id_t>();

        std::vector<vertex_id_t> local_vertex_degree(v_num, 0);

        for (edge_id_t e_i = 0; e_i < read_e_num; e_i++) 
        {

            local_vertex_degree[read_edges[e_i].src]++;
        }

        MPI_Allreduce(local_vertex_degree.data(),  vertex_out_degree, v_num, get_mpi_data_type<vertex_id_t>(), MPI_SUM, MPI_COMM_WORLD);
        
        vertex_id_t degree_zero_n=0;
        for(vertex_id_t v_i = 0; v_i <v_num;v_i++){
            if(vertex_out_degree[v_i]==0){
                degree_zero_n ++;
            }
        }
        printf("[p%d] degree zero %u\n",get_mpi_rank(),degree_zero_n);

        std::fill(local_vertex_degree.begin(), local_vertex_degree.end(), 0);

        for (edge_id_t e_i = 0; e_i < read_e_num; e_i++) 
        {
            local_vertex_degree[read_edges[e_i].dst] ++;
        }

        MPI_Allreduce(local_vertex_degree.data(),  vertex_in_degree, v_num, get_mpi_data_type<vertex_id_t>(), MPI_SUM, MPI_COMM_WORLD);


        vertex_partition_begin = new vertex_id_t[partition_num];
        vertex_partition_end = new vertex_id_t[partition_num];

        edge_id_t total_workload = 0;
        this->e_num = 0;
        for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {

            total_workload += V_alpha + vertex_out_degree[v_i];
            
            this->e_num += vertex_out_degree[v_i];
        }
        
        edge_id_t workload_per_node = (total_workload + partition_num - 1) / partition_num;
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            if (p_i == 0)
            {
                this->vertex_partition_begin[p_i] = 0;
            } else
            {
                this->vertex_partition_begin[p_i] = this->vertex_partition_end[p_i - 1];
            }

            this->vertex_partition_end[p_i] = this->vertex_partition_begin[p_i];

            edge_id_t workload = 0;
            for (vertex_id_t v_i = vertex_partition_begin[p_i]; v_i < v_num && workload < workload_per_node; v_i++)
            {
                workload += V_alpha + vertex_out_degree[v_i];
                vertex_partition_end[p_i]++;
            }
            
#ifdef PERF_PROF
            if (this->local_partition_id == 0)
            {
                std::cout << "[partition " << static_cast<int>(p_i) << ":] (Start " << vertex_partition_begin[p_i] << " End: " << vertex_partition_end[p_i] 
                << ") (Workload: " << workload << "Each machine Workload: " << workload_per_node << " ) "<< " Vertex Num " << vertex_partition_end[p_i] - vertex_partition_begin[p_i] << std::endl;
                // printf("partition %d: %u %u (%zu %zu)\n", p_i, vertex_partition_begin[p_i], vertex_partition_end[p_i], workload, workload_per_node);
            }
#endif
        }

        if(string("")!= string(partition_path))
        {
            FILE* fp = fopen(partition_path,"r");
            assert(fp!= nullptr);
            for(int p_i=0;p_i<partition_num; p_i++)
            {
                int t = fscanf(fp,"%u %u\n",&vertex_partition_begin[p_i],&vertex_partition_end[p_i]);
                assert(t == 2);
            }
            fclose(fp);
        }


        assert(this->vertex_partition_end[partition_num - 1] == v_num);

        this->vertex_partition_id = alloc_vertex_array<partition_id_t>();
        for (partition_id_t p_i = 0; p_i < this->partition_num; p_i++)
        {
            for (vertex_id_t v_i = this->vertex_partition_begin[p_i]; v_i < this->vertex_partition_end[p_i]; v_i++)
            {
                this->vertex_partition_id[v_i] = p_i;
            }
        }
       
        this->local_e_num = 0;
        for (vertex_id_t v_i = vertex_partition_begin[local_partition_id]; v_i < vertex_partition_end[local_partition_id]; v_i++)
        {
            this->local_e_num += this->vertex_out_degree[v_i];
        }
        printf("[partition %d]begin:%u end:%u num:%u degree:%lu\n",
        static_cast<int>(this->local_partition_id),vertex_partition_begin[local_partition_id],vertex_partition_end[local_partition_id],
        vertex_partition_end[local_partition_id]-vertex_partition_begin[local_partition_id],local_e_num);

        Edge<edge_data_t> *local_edges = new Edge<edge_data_t>[local_e_num];

        shuffle_edges(read_edges, read_e_num, local_edges, local_e_num);

        this -> csr = new EdgeContainer<edge_data_t>();
        this-> co_occor = new CoOccorCsr();
        build_edge_container(local_edges, local_e_num, this->csr, vertex_out_degree);
        // printf("co_coccor build before====ok\n");
         //g_build_edge_container(g_read_edges,g_read_e_num,v_num,this->g_csr,vertex_out_degree);
        g_build_edge_container(local_edges,local_e_num,vertex_out_degree);
        // printf("co_coor build====ok\n");
        // printEdgeContainer(this->csr->adj_lists);

        delete []read_edges;
        delete []local_edges;
        
        send_locks = new std::mutex[partition_num];
        recv_locks = new std::mutex[partition_num];

        dist_exec_ctx.progress = new size_t*[worker_num];
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            dist_exec_ctx.progress[t_i] = new size_t[partition_num];
        }

#ifdef PERF_PROF
        printf("finish build graph, time %.3lfs\n", timer.duration());
#endif
    }

    void load_commonNeighbors(const char* fname)
    {
        Timer timer;
        // FILE *f = fopen(fname, "r");
        // assert(f != NULL);
        // fseek(f, 0, SEEK_SET);
        // int src , dst, comNeibors;
        // // if(this->commonNeighbors.size() == 0) { this->commonNeighbors = new un}
        // while(3 == fscanf(f, "%u %u %u", &src, &dst, &comNeibors))
        // {
        //     //printf("%d %d %d\n", src, dst, comNeibors);
        //     // assert(commonNeighbors.find({src, dst}) == commonNeighbors.end());
        //     // assert(commonNeighbors.find({dst, src}) == commonNeighbors.end());
        //     commonNeighbors[{src,dst}] = comNeibors;
        //     commonNeighbors[{dst,src}] = comNeibors;
        // } 
        // Edge<edge_data_t> *read_edges;
        // read_graph_common_neighbour(fname, read_edges, this->commonNeighbors, this->vertex_partition_id, this->local_partition_id);
        // assert(this->commonNeighbors.size() == this->local_e_num);
        // // test 
        // // 9918 10057 43
        // // 9918 10109 64 
        // // U u1 = {9918, 10057};
        // // U u2 = {9918, 10109};
        // // std::cout << this->commonNeighbors[{9918,10057}] << " " << this->commonNeighbors[{9918,10109}] << std::endl;
        // //  std::cout << this->commonNeighbors[u1.key] << " " << this->commonNeighbors[u2.key] << std::endl;
        // delete []read_edges;
#ifdef PERF_PROF
        printf("finish common neighbour, time %.3lfs\n", timer.duration());
        // exit(0);
#endif   
    }

    void set_msg_buffer(size_t max_msg_num, size_t max_msg_size)
    {
        if (thread_local_msg_buffer == nullptr)
        {
            thread_local_msg_buffer = new MessageBuffer*[worker_num];
            #pragma omp parallel
            {
                int worker_id = omp_get_thread_num();
                thread_local_msg_buffer[worker_id] = new MessageBuffer();
            }
        }
        if (msg_send_buffer == nullptr)
        {
            msg_send_buffer = new MessageBuffer*[partition_num];
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_send_buffer[p_i] = new MessageBuffer();
            }
        }
        if (msg_recv_buffer == nullptr)
        {
            msg_recv_buffer = new MessageBuffer*[partition_num];
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_recv_buffer[p_i] = new MessageBuffer();
            }
        }

        size_t local_buf_size = max_msg_size * THREAD_LOCAL_BUF_CAPACITY;
        #pragma omp parallel
        {
            int worker_id = omp_get_thread_num();
            if (thread_local_msg_buffer[worker_id]->sz < local_buf_size)
            {
                thread_local_msg_buffer[worker_id]->alloc(local_buf_size);
            }
        }
        size_t comm_buf_size = max_msg_size * max_msg_num;
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            if (msg_send_buffer[p_i]->sz < comm_buf_size)
            {
                msg_send_buffer[p_i]->alloc(comm_buf_size);
            }
            if (msg_recv_buffer[p_i]->sz < comm_buf_size)
            {
                msg_recv_buffer[p_i]->alloc(comm_buf_size);
            }
        }
    }

    void free_msg_buffer()
    {
        if (thread_local_msg_buffer != nullptr)
        {
            for (partition_id_t t_i = 0; t_i < worker_num; t_i ++)
            {
                delete thread_local_msg_buffer[t_i];
            }
            delete []thread_local_msg_buffer;
        }
        if (msg_send_buffer != nullptr)
        {
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                delete msg_send_buffer[p_i];
            }
            delete []msg_send_buffer;
        }
        if (msg_recv_buffer != nullptr)
        {
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                delete msg_recv_buffer[p_i];
            }
            delete []msg_recv_buffer;
        }
    }

    template<typename msg_data_t>
    void emit(vertex_id_t dst_id, msg_data_t data, int worker_id)
    {
        
        typedef Message<msg_data_t> msg_t;
        msg_t* buf_data = (msg_t*)thread_local_msg_buffer[worker_id]->data;
        
        auto &count = thread_local_msg_buffer[worker_id]->count;
        
        buf_data[count].dst_vertex_id = dst_id;
        buf_data[count].data = data;
        count++;
#ifdef UNIT_TEST
        thread_local_msg_buffer[worker_id]->self_check<msg_t>();
        assert(dst_id < v_num);
#endif
        
        if (count == THREAD_LOCAL_BUF_CAPACITY)
        {
            flush_thread_local_msg_buffer<msg_t>(worker_id);
        }
    }


    template<typename msg_data_t>
    void emit(vertex_id_t dst_id, msg_data_t data)
    {
        emit(dst_id, data, omp_get_thread_num());
    }

    template<typename msg_t>
    void flush_thread_local_msg_buffer(partition_id_t worker_id)
    {
        
        auto local_buf = this->thread_local_msg_buffer[worker_id];
        
        msg_t *local_data = (msg_t*)local_buf->data;
        
        auto &local_msg_count = local_buf->count;
        
        if (local_msg_count != 0)
        {
            vertex_id_t dst_count[partition_num];
            std::fill(dst_count, dst_count + partition_num, 0);
            
            for (vertex_id_t m_i = 0; m_i < local_msg_count; m_i++)
            {
                dst_count[vertex_partition_id[local_data[m_i].dst_vertex_id]] ++;
            }
    
            msg_t *dst_data_pos[partition_num];
            size_t end_data_pos[partition_num];
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                
                vertex_id_t start_pos = __sync_fetch_and_add(&this->msg_send_buffer[p_i]->count, dst_count[p_i]);
#ifdef UNIT_TEST
                msg_send_buffer[p_i]->self_check<msg_t>();
#endif

                dst_data_pos[p_i] = (msg_t*)(this->msg_send_buffer[p_i]->data) + start_pos;
      
                end_data_pos[p_i] = start_pos + dst_count[p_i];
            }
   
            for (vertex_id_t m_i = 0; m_i < local_msg_count; m_i++)
            {
                *(dst_data_pos[vertex_partition_id[local_data[m_i].dst_vertex_id]]++) =  local_data[m_i];
            }
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                dist_exec_ctx.progress[worker_id][p_i] = end_data_pos[p_i];
            }
            local_msg_count = 0;
        }
    }

    void notify_progress(vertex_id_t progress_begin, vertex_id_t progress_end, vertex_id_t workload, bool phased_exec)
    {
        int phase_num = phased_exec ? DISTRIBUTEDEXECUTIONCTX_PHASENUM : 1;
        if (phase_num > 1)
        {
            vertex_id_t work_per_phase = workload / phase_num + 1;
            int phase_begin = 0;
            while (progress_begin >= work_per_phase)
            {
                phase_begin ++;
                progress_begin -= work_per_phase;
            }
            int phase_end = 0;
            while (progress_end >= work_per_phase)
            {
                phase_end ++;
                progress_end -= work_per_phase;
            }
            if (phase_end == phase_num)
            {
                phase_end --;
            }
            for (int phase_i = phase_begin; phase_i < phase_end; phase_i++)
            {
                dist_exec_ctx.phase_locks[phase_i].unlock();
                __sync_fetch_and_add(&dist_exec_ctx.unlocked_phase, 1);
            }
        }
    }

    template<typename msg_data_t>
    size_t distributed_execute(
        std::function<void(void)> msg_producer,
        std::function<void(Message<msg_data_t> *, Message<msg_data_t> *)> msg_consumer,
        Message<msg_data_t> *zero_copy_data = nullptr,
        bool phased_exec = false
    )
    {
        typedef Message<msg_data_t> msg_t;
        Timer timer;

        int phase_num = phased_exec ? DISTRIBUTEDEXECUTIONCTX_PHASENUM : 1;
        for (int phase_i = 0; phase_i < phase_num; phase_i++)
        {
            dist_exec_ctx.phase_locks[phase_i].lock();
        }


        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                dist_exec_ctx.progress[t_i][p_i] = 0;
            }
        }
        dist_exec_ctx.unlocked_phase = 0;
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            recv_locks[p_i].lock();
        }
        volatile size_t zero_copy_recv_count = 0;
        

        std::thread recv_thread([&](){
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_recv_buffer[p_i]->count = 0;
            }
            std::vector<MPI_Request*> requests[partition_num];
            auto recv_func = [&] (partition_id_t src)
            {
                MPI_Status prob_status;
                MPI_Probe(src, Tag_Msg, MPI_COMM_WORLD, &prob_status);
                int sz;
                MPI_Get_count(&prob_status, get_mpi_data_type<char>(), &sz);
                // printf("recv %u <- %u: %zu\n", local_partition_id, src, sz / sizeof(msg_t));
                MPI_Request *recv_req = new MPI_Request();
                requests[src].push_back(recv_req);
                if (zero_copy_data == nullptr)
                {
                    MPI_Irecv(((msg_t*)msg_recv_buffer[src]->data) + msg_recv_buffer[src]->count, sz, get_mpi_data_type<char>(), src, Tag_Msg, MPI_COMM_WORLD, recv_req);
                    msg_recv_buffer[src]->count += sz / sizeof(msg_t);
                    msg_recv_buffer[src]->template self_check<msg_t>();
                } else
                {
                    MPI_Irecv(zero_copy_data + zero_copy_recv_count, sz, get_mpi_data_type<char>(), src, Tag_Msg, MPI_COMM_WORLD, recv_req);
                    zero_copy_recv_count += sz / sizeof(msg_t);
                }
               
            };
            for (int phase_i = 0; phase_i < phase_num; phase_i ++)
            {
                if (phase_i + 1 == phase_num)
                {
                    for (partition_id_t step = 0; step < partition_num; step++)
                    {
                        partition_id_t src = (partition_num + local_partition_id - step) % partition_num;
                        recv_func(src);
                    }
                } else
                {
                    partition_id_t src = (partition_num + local_partition_id - phase_i % partition_num) % partition_num;
                    recv_func(src);
                }
            }
            for (partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t src = (partition_num + local_partition_id - step) % partition_num;
                for (auto req : requests[src])
                {
                    MPI_Status status;
                    MPI_Wait(req, &status);
                    delete req;
                }
                recv_locks[src].unlock();
            }
        });

        std::thread send_thread([&](){
            size_t send_progress[partition_num];    
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                send_progress[p_i] = 0;
            }
            std::vector<MPI_Request*> requests;
            auto send_func = [&] (partition_id_t dst, size_t diff)
            {
                msg_send_buffer[dst]->template self_check<msg_t>();
                MPI_Request* req = new MPI_Request();
                requests.push_back(req);
                // std::cout << "diff * sizeof(msg_t) = " << diff << " " << sizeof(msg_t) << std::endl;
                assert(diff * sizeof(msg_t) < INT_MAX);
        
                MPI_Isend(((msg_t*)msg_send_buffer[dst]->data) + send_progress[dst], diff * sizeof(msg_t), get_mpi_data_type<char>(), dst, Tag_Msg, MPI_COMM_WORLD, req);
#ifdef PERF_PROF
                if (local_partition_id == 0)
                {
                    printf("end send %u -> %u: %zu time %lf\n", local_partition_id, dst, diff, timer.duration());
                }
#endif
                send_progress[dst] += diff;
            };
            for (int phase_i = 0; phase_i < phase_num; phase_i++)
            {
                dist_exec_ctx.phase_locks[phase_i].lock();

                if (phase_i + 1 == phase_num)
                {
                    for (partition_id_t step = 0; step < partition_num; step++)
                    {
                        
                        partition_id_t dst = (local_partition_id + step) % partition_num;
                        size_t max_progress = 0;
                        
                        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
                        {
                            volatile size_t temp_val = dist_exec_ctx.progress[t_i][dst];
                            if (temp_val > max_progress)
                            {
                                max_progress = temp_val;
                            }
                        }
                        size_t diff = max_progress - send_progress[dst];
                        send_func(dst, diff);
                    }
                } else
                {
                    partition_id_t dst = (local_partition_id + phase_i) % partition_num;
                    size_t min_progress = UINT_MAX;
                    for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
                    {
                        volatile size_t temp_val = dist_exec_ctx.progress[t_i][dst];
                        if (temp_val < min_progress)
                        {
                            min_progress = temp_val;
                        }
                    }
                    size_t diff = min_progress - send_progress[dst];
                    send_func(dst, diff);
                }
                dist_exec_ctx.phase_locks[phase_i].unlock();
            }
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_send_buffer[p_i]->count = 0;
            }
            for (auto req : requests)
            {
                MPI_Status status;
                MPI_Wait(req, &status);
                delete req;
            }
        });

        msg_producer();

        size_t flush_workload = 0;
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            flush_workload += thread_local_msg_buffer[t_i]->count;
        }
#pragma omp parallel for if (flush_workload * 2 >= OMP_PARALLEL_THRESHOLD)
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            flush_thread_local_msg_buffer<msg_t>(t_i);
        }

        for (int phase_i = dist_exec_ctx.unlocked_phase; phase_i < phase_num; phase_i++)
        {
            dist_exec_ctx.phase_locks[phase_i].unlock();
        }
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish msg_producer in %lfs\n", local_partition_id, timer.duration());
        }
#endif

        size_t msg_num = 0;
        for (int step = 0; step < partition_num; step++)
        {
            partition_id_t src_partition_id = (partition_num + local_partition_id - step) % partition_num;
            recv_locks[src_partition_id].lock();
            if (zero_copy_data == nullptr)
            {
                size_t data_amount = msg_recv_buffer[src_partition_id]->count;
                msg_num += data_amount;
                msg_t* data_begin = (msg_t*)(msg_recv_buffer[src_partition_id]->data);
                msg_t* data_end = data_begin + data_amount;
                msg_consumer(data_begin, data_end);
            }
            recv_locks[src_partition_id].unlock();
        }
        if (zero_copy_data != nullptr)
        {
            msg_consumer(zero_copy_data, zero_copy_data + zero_copy_recv_count);
            msg_num = zero_copy_recv_count;
        }
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish msg_consumer in %lfs\n", local_partition_id, timer.duration());
        }
#endif
        recv_thread.join();
        send_thread.join();
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish transmission in %lfs\n", local_partition_id, timer.duration());
        }
#endif
        size_t glb_msg_num;
        MPI_Allreduce(&msg_num, &glb_msg_num, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        return glb_msg_num;
    }

    template<typename msg_data_t>
    size_t distributed_execute_init_walkers(
        std::function<void(void)> msg_producer,
        std::function<void(Message<msg_data_t> *, Message<msg_data_t> *)> msg_consumer,
        Message<msg_data_t> *zero_copy_data = nullptr,
        bool phased_exec = false
    )
    {
        typedef Message<msg_data_t> msg_t;
        Timer timer;

        int phase_num = phased_exec ? DISTRIBUTEDEXECUTIONCTX_PHASENUM : 1;
        for (int phase_i = 0; phase_i < phase_num; phase_i++)
        {
            dist_exec_ctx.phase_locks[phase_i].lock();
        }


        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                dist_exec_ctx.progress[t_i][p_i] = 0;
            }
        }
        dist_exec_ctx.unlocked_phase = 0;
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            recv_locks[p_i].lock();
        }
        volatile size_t zero_copy_recv_count = 0;
        

        std::thread recv_thread([&](){
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_recv_buffer[p_i]->count = 0;
            }
            std::vector<MPI_Request*> requests[partition_num];
            auto recv_func = [&] (partition_id_t src)
            {
                MPI_Status prob_status0;
                MPI_Probe(src, Tag_Msg_Count, MPI_COMM_WORLD, &prob_status0);
                int sz0;
                MPI_Get_count(&prob_status0, get_mpi_data_type<char>(), &sz0);
                
                MPI_Request *recv_req0 = new MPI_Request();
                requests[src].push_back(recv_req0);
                int count = 0;
                MPI_Irecv(&count, 1 , get_mpi_data_type<int>(), src, Tag_Msg_Count, MPI_COMM_WORLD, recv_req0);
                printf("received%d\n", count);

                for(int i = 0; i < count; i++){
                    MPI_Status prob_status;
                    MPI_Probe(src, Tag_Msg, MPI_COMM_WORLD, &prob_status);
                    int sz;
                    MPI_Get_count(&prob_status, get_mpi_data_type<char>(), &sz);
                    printf("recv %u <- %u: %zu\n", local_partition_id, src, sz / sizeof(msg_t));
                    MPI_Request *recv_req = new MPI_Request();
                    requests[src].push_back(recv_req);
                    if (zero_copy_data == nullptr)
                    {
                        MPI_Irecv(((msg_t*)msg_recv_buffer[src]->data) + msg_recv_buffer[src]->count, sz, get_mpi_data_type<char>(), src, Tag_Msg, MPI_COMM_WORLD, recv_req);
                        msg_recv_buffer[src]->count += sz / sizeof(msg_t);
                        msg_recv_buffer[src]->template self_check<msg_t>();
                    } else
                    {
                        MPI_Irecv(zero_copy_data + zero_copy_recv_count, sz, get_mpi_data_type<char>(), src, Tag_Msg, MPI_COMM_WORLD, recv_req);
                        zero_copy_recv_count += sz / sizeof(msg_t);
                    }
                }
               
            };
            for (int phase_i = 0; phase_i < phase_num; phase_i ++)
            {
                if (phase_i + 1 == phase_num)
                {
                    for (partition_id_t step = 0; step < partition_num; step++)
                    {
                        partition_id_t src = (partition_num + local_partition_id - step) % partition_num;
                        recv_func(src);
                    }
                } else
                {
                    partition_id_t src = (partition_num + local_partition_id - phase_i % partition_num) % partition_num;
                    recv_func(src);
                }
            }
            for (partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t src = (partition_num + local_partition_id - step) % partition_num;
                for (auto req : requests[src])
                {
                    MPI_Status status;
                    MPI_Wait(req, &status);
                    delete req;
                }
                recv_locks[src].unlock();
            }
        });

        std::thread send_thread([&](){
            size_t send_progress[partition_num];    
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                send_progress[p_i] = 0;
            }
            std::vector<MPI_Request*> requests;
            auto send_func = [&] (partition_id_t dst, size_t diff)
            {
                msg_send_buffer[dst]->template self_check<msg_t>();
                // MPI_Request* req = new MPI_Request();
                // requests.push_back(req);
                std::cout << "diff * sizeof(msg_t) = " << diff << " " << sizeof(msg_t) << std::endl;
                // assert(diff * sizeof(msg_t) < INT_MAX);
                size_t send_data_size = diff * sizeof(msg_t);
                if(send_data_size > INT_MAX){
                    int count = 3;
                    MPI_Request* req0 = new MPI_Request();
                    requests.push_back(req0);
                    MPI_Isend(&count, 1, get_mpi_data_type<int>(), dst, Tag_Msg_Count, MPI_COMM_WORLD, req0);
                    std::cout << "Send" << count << "\n";

                    MPI_Request* req = new MPI_Request();
                    requests.push_back(req);
                    int msg_num = 15000000;
                    MPI_Isend(((msg_t*)msg_send_buffer[dst]->data) + send_progress[dst], (diff-2*msg_num) * sizeof(msg_t), get_mpi_data_type<char>(), dst, Tag_Msg, MPI_COMM_WORLD, req);
                    send_progress[dst] += (diff-2*msg_num);
                    MPI_Request* req1 = new MPI_Request();
                    requests.push_back(req1);
                    MPI_Isend(((msg_t*)msg_send_buffer[dst]->data) + send_progress[dst], (msg_num) * sizeof(msg_t), get_mpi_data_type<char>(), dst, Tag_Msg, MPI_COMM_WORLD, req1);
                    send_progress[dst] += (msg_num);
                    MPI_Request* req2 = new MPI_Request();
                    requests.push_back(req2);
                    MPI_Isend(((msg_t*)msg_send_buffer[dst]->data) + send_progress[dst], (msg_num) * sizeof(msg_t), get_mpi_data_type<char>(), dst, Tag_Msg, MPI_COMM_WORLD, req2);
                    send_progress[dst] += (msg_num);
                    
                }else{
                    int count = 1;
                    MPI_Request* req0 = new MPI_Request();
                    requests.push_back(req0);
                    MPI_Isend(&count, 1, get_mpi_data_type<int>(), dst, Tag_Msg_Count, MPI_COMM_WORLD, req0);
                    std::cout << "Send" << count << "\n";
                    
                    MPI_Request* req = new MPI_Request();
                    requests.push_back(req);
                    MPI_Isend(((msg_t*)msg_send_buffer[dst]->data) + send_progress[dst], diff * sizeof(msg_t), get_mpi_data_type<char>(), dst, Tag_Msg, MPI_COMM_WORLD, req);
                    send_progress[dst] += diff;
                }
                // MPI_Isend(((msg_t*)msg_send_buffer[dst]->data) + send_progress[dst], diff * sizeof(msg_t), get_mpi_data_type<char>(), dst, Tag_Msg, MPI_COMM_WORLD, req);
#ifdef PERF_PROF
                if (local_partition_id == 0)
                {
                    printf("end send %u -> %u: %zu time %lf\n", local_partition_id, dst, diff, timer.duration());
                }
#endif
                // send_progress[dst] += diff;
            };
            for (int phase_i = 0; phase_i < phase_num; phase_i++)
            {
                dist_exec_ctx.phase_locks[phase_i].lock();

                if (phase_i + 1 == phase_num)
                {
                    for (partition_id_t step = 0; step < partition_num; step++)
                    {
   
                        partition_id_t dst = (local_partition_id + step) % partition_num;
                        size_t max_progress = 0;

                        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
                        {
                            volatile size_t temp_val = dist_exec_ctx.progress[t_i][dst];
                            if (temp_val > max_progress)
                            {
                                max_progress = temp_val;
                            }
                        }
                        std::cout << "max_progress = " << max_progress << std::endl;
                        size_t diff = max_progress - send_progress[dst];
                        send_func(dst, diff);
                    }
                } else
                {
                    partition_id_t dst = (local_partition_id + phase_i) % partition_num;
                    size_t min_progress = UINT_MAX;
                    for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
                    {
                        volatile size_t temp_val = dist_exec_ctx.progress[t_i][dst];
                        if (temp_val < min_progress)
                        {
                            min_progress = temp_val;
                        }
                    }
                    std::cout << "min_progress = " << min_progress << std::endl;
                    size_t diff = min_progress - send_progress[dst];
                    send_func(dst, diff);
                }
                dist_exec_ctx.phase_locks[phase_i].unlock();
            }
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_send_buffer[p_i]->count = 0;
            }
            for (auto req : requests)
            {
                MPI_Status status;
                MPI_Wait(req, &status);
                delete req;
            }
        });

        msg_producer();

        size_t flush_workload = 0;
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            flush_workload += thread_local_msg_buffer[t_i]->count;
        }
#pragma omp parallel for if (flush_workload * 2 >= OMP_PARALLEL_THRESHOLD)
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            flush_thread_local_msg_buffer<msg_t>(t_i);
        }

        for (int phase_i = dist_exec_ctx.unlocked_phase; phase_i < phase_num; phase_i++)
        {
            dist_exec_ctx.phase_locks[phase_i].unlock();
        }
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish msg_producer in %lfs\n", local_partition_id, timer.duration());
        }
#endif

        size_t msg_num = 0;
        for (int step = 0; step < partition_num; step++)
        {
            partition_id_t src_partition_id = (partition_num + local_partition_id - step) % partition_num;
            recv_locks[src_partition_id].lock();
            if (zero_copy_data == nullptr)
            {
                size_t data_amount = msg_recv_buffer[src_partition_id]->count;
                msg_num += data_amount;
                msg_t* data_begin = (msg_t*)(msg_recv_buffer[src_partition_id]->data);
                msg_t* data_end = data_begin + data_amount;
                msg_consumer(data_begin, data_end);
            }
            recv_locks[src_partition_id].unlock();
        }
        if (zero_copy_data != nullptr)
        {
            msg_consumer(zero_copy_data, zero_copy_data + zero_copy_recv_count);
            msg_num = zero_copy_recv_count;
        }
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish msg_consumer in %lfs\n", local_partition_id, timer.duration());
        }
#endif
        recv_thread.join();
        send_thread.join();
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish transmission in %lfs\n", local_partition_id, timer.duration());
        }
#endif
        size_t glb_msg_num;
        MPI_Allreduce(&msg_num, &glb_msg_num, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        return glb_msg_num;
    }
};
