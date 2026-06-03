// SRHT + k-bit scalar quantization: RaBitQ rotation insight + multi-bit accuracy
#include "common_includes.h"
#include <boost/program_options.hpp>
#include <algorithm>
#include <random>
#include "utils.h"
#include "rabitq.h"
namespace po = boost::program_options;
using namespace diskann;

// Hadamard (power-of-2 D only)
static void hadamard(float *v, uint32_t D){
    for(uint32_t h=1;h<D;h<<=1)
        for(uint32_t i=0;i<D;i+=h<<1)
            for(uint32_t j=i;j<i+h;j++){float a=v[j],b=v[j+h];v[j]=a+b;v[j+h]=a-b;}
    float s=1.f/sqrtf((float)D);
    for(uint32_t i=0;i<D;i++) v[i]*=s;
}
static void srht(float *v, const int8_t *signs, uint32_t D){
    for(uint32_t i=0;i<D;i++) v[i]*=(float)signs[i];
    hadamard(v,D);
}

int build_sqk(const std::string &base_path, const std::string &out_prefix,
              uint32_t k, bool use_srht)
{
    float *base=nullptr; size_t N,D,adim;
    diskann::load_aligned_bin<float>(base_path,base,N,D,adim);
    diskann::cout<<"Loaded "<<N<<" x "<<D<<" k="<<k<<" srht="<<use_srht<<std::endl;

    // Per-dim means (centering)
    std::vector<double> cd(D,0.0);
    for(size_t i=0;i<N;i++){const float*x=base+i*adim; for(uint32_t d=0;d<D;d++) cd[d]+=x[d];}
    std::vector<float> centers(D);
    for(uint32_t d=0;d<D;d++) centers[d]=(float)(cd[d]/(double)N);

    // Random signs for SRHT (fixed seed)
    std::mt19937 rng(42);
    std::vector<int8_t> signs(D);
    for(uint32_t d=0;d<D;d++) signs[d]=(rng()&1)?1:-1;

    // Build working buffer: apply SRHT if requested, then collect per-dim stats
    std::vector<float> work(D);
    uint32_t clip=(uint32_t)(N*0.01);
    std::vector<float> col(N);
    std::vector<float> mins(D), scales(D);

    for(uint32_t d=0;d<D;d++){
        for(size_t i=0;i<N;i++){
            const float*x=base+i*adim;
            // center
            for(uint32_t dd=0;dd<D;dd++) work[dd]=x[dd]-centers[dd];
            if(use_srht) srht(work.data(),signs.data(),(uint32_t)D);
            col[i]=work[d];
        }
        std::sort(col.begin(),col.end());
        float lo=col[clip], hi=col[N-1-clip];
        if(hi<=lo) hi=lo+1e-6f;
        mins[d]=lo; scales[d]=(hi-lo)/((float)((1u<<k)-1));
    }
    diskann::cout<<"Per-dim stats computed."<<std::endl;

    // Encode
    uint32_t bpv=sqk_bytes_per_vec((uint32_t)D,k);
    std::vector<uint8_t> codes(N*bpv,0);
    uint32_t levels=(1u<<k)-1;
    for(size_t i=0;i<N;i++){
        const float*x=base+i*adim;
        for(uint32_t dd=0;dd<D;dd++) work[dd]=x[dd]-centers[dd];
        if(use_srht) srht(work.data(),signs.data(),(uint32_t)D);
        uint8_t*c=codes.data()+i*bpv;
        for(uint32_t d=0;d<D;d++){
            uint32_t q=(uint32_t)std::max(0,(int)std::round((work[d]-mins[d])/scales[d]));
            if(q>levels)q=levels;
            if(k==4){if(d&1)c[d>>1]|=(uint8_t)(q<<4);else c[d>>1]=(uint8_t)(q&0xF);}
            else{// k==6
                uint32_t bo=d*6,bi=bo>>3,sh=bo&7;
                c[bi]|=(uint8_t)((q<<sh)&0xFF);
                if(sh+6>8)c[bi+1]|=(uint8_t)(q>>(8-sh));
            }
        }
    }
    delete[] base;

    // Write: header + centers + signs(if srht) + mins + scales + codes
    std::string out=out_prefix+"_sqk.bin";
    std::ofstream f(out,std::ios::binary);
    uint64_t hdr[4]={(uint64_t)N,(uint64_t)D,(uint64_t)k,(uint64_t)(use_srht?1:0)};
    f.write((char*)hdr,sizeof(hdr));
    f.write((char*)centers.data(),D*4);
    if(use_srht) f.write((char*)signs.data(),D);
    f.write((char*)mins.data(),D*4);
    f.write((char*)scales.data(),D*4);
    f.write((char*)codes.data(),N*bpv);
    f.close();
    double mb=(4.0*8+D*4+(use_srht?D:0)+D*8+(double)N*bpv)/1048576.0;
    diskann::cout<<"Written "<<out<<"  ("<<mb<<" MB)"<<std::endl;
    return 0;
}

int main(int argc,char **argv){
    std::string base,prefix; uint32_t k; bool srht;
    po::options_description desc("SRHT + k-bit scalar quantization (RaBitQ adaptation)");
    desc.add_options()
        ("help,h","help")
        ("base_path,b",po::value<std::string>(&base)->required(),"base .fbin")
        ("index_path_prefix,i",po::value<std::string>(&prefix)->required(),"prefix")
        ("bits,k",po::value<uint32_t>(&k)->default_value(4),"bits per dim (4 or 6)")
        ("srht,r",po::value<bool>(&srht)->default_value(true),"apply SRHT rotation (0/1)");
    po::variables_map vm;
    try{po::store(po::parse_command_line(argc,argv,desc),vm);
        if(vm.count("help")){diskann::cout<<desc;return 0;}
        po::notify(vm);}
    catch(const std::exception &e){std::cerr<<e.what()<<"\n";return -1;}
    if(k!=4&&k!=6){std::cerr<<"Only k=4 or k=6\n";return -1;}
    return build_sqk(base,prefix,k,srht);
}
