#pragma once
#include "type.hpp"
#include <cstddef>
#include <iostream>
#include <vector>

using namespace std;
using corpus_t = vector<vector<vertex_id_t>>;
class Bitmap {
private:
    size_t bitSize = 0;
    const int EXPANDSIZE = 1;
    // 获取位所在的字节索引
    size_t getByteIndex(size_t bitIndex) const {
        return bitIndex / 8;
    }
    // 获取位在字节中的位置
    size_t getBitOffset(size_t bitIndex) const {
        return bitIndex % 8;
    }
    // 扩展内存以容貌指定的位
    void expandToFit(size_t bitIndex) {
        size_t requiredBytes = getByteIndex(bitIndex) + 1;
        if(data.size()<requiredBytes){
            data.resize(requiredBytes + EXPANDSIZE, 0);
        }
    }
public:
    vector<char> data;
    Bitmap(){
        data.resize(2,0);
    }
    Bitmap(size_t bitNum){
        data.resize(bitNum/ sizeof(char) +1);
    }
    // 设置指定位位1
    void set(size_t bitIndex) {
        expandToFit(bitIndex);
        size_t byteIndex = getByteIndex(bitIndex);
        size_t bitOffset = getBitOffset(bitIndex);
        data[byteIndex] |= (1 << bitOffset);
        this->bitSize = max(bitSize,bitIndex+1);
    }

    void unset(size_t bitIndex) {
        expandToFit(bitIndex);
        size_t byteIndex = getByteIndex(bitIndex);
        size_t bitOffset = getBitOffset(bitIndex);
        data[byteIndex] &= ~(1 << bitOffset);
        this->bitSize = max(bitSize,bitIndex + 1);
    }

    // 检查指定位是否为1
    bool check(size_t bitIndex)const {
        size_t byteIndex = getByteIndex(bitIndex);
        size_t bitOffset = getBitOffset(bitIndex);
        if(byteIndex >= data.size()){
            cout << "bitmap out of range\n";
            return false;
        }
        return (data[byteIndex] & (1 << bitOffset)) != 0;
    }
    
    size_t size() {
        return this-> bitSize;
    }

    void printData(){
        for(size_t i = 0; i < data.size() * 8; i++){
            cout<< check(i);
        }
        cout << endl;
    }
};

class HeadMapSequence {
public:
    Bitmap headmap;
    vector<vertex_id_t> misc_data;
};

using compress_t = vector<HeadMapSequence>; 

class CorpusCompressor {
public:
    void compressSequence(vector<vertex_id_t> &seq,HeadMapSequence& hms) {
        vertex_id_t headNode = seq[0];
        hms.misc_data.push_back(headNode);
        for(size_t  i = 1; i < seq.size(); i++) {
            if(seq[i] == headNode){
                hms.headmap.set(i);
                continue;
            }else{
                hms.misc_data.push_back(seq[i]);
            }
        }
    }

    void uncompressSequence(vector<vertex_id_t> &seq,HeadMapSequence& hms) {
        vertex_id_t headNode = hms.misc_data[0];
        int p = 0;// p misc_data point
        int q = 0; // map point
        while(q < hms.headmap.size()){
            if(hms.headmap.check(q)){
                seq.push_back(headNode);
            }else {
                seq.push_back(hms.misc_data[p]);
                p++;
            }
            q++;
        }
        while(p < hms.misc_data.size()){
            seq.push_back(hms.misc_data[p]);
            p++;
        }
    }

    void compressCorpus(corpus_t &cor, compress_t &cp) {
        for(size_t i = 0; i < cor.size(); i++){
            HeadMapSequence hms;
            compressSequence(cor[i], hms);
            cp.push_back(hms);
        }
    }

    void uncompressCorpus(corpus_t &cor,compress_t& cp) {
        for(size_t i = 0; i < cp.size(); i++){
            vector<vertex_id_t> seq;
            uncompressSequence(seq, cp[i]);
            cor.push_back(seq);
        }
    }

    void printCorpus(vector<vector<vertex_id_t>> &corpus){
        cout << "=== corpus print === " << endl;
        for(size_t i = 0; i < corpus.size(); i++){
            for(size_t j = 0; j < corpus[i].size(); j++){
                cout << corpus[i][j] <<" ";
            }
            cout << endl;
        }
        cout << "====================" << endl;
    }
};
