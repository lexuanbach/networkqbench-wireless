// Exact double-complex density-matrix evolution for rz/sx/x/cx with CX depolarization.
// No truncation, trajectory approximation, or sparsification is used.
#include <complex>
#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
using C=std::complex<double>;
extern "C" int density_probabilities(int n,int count,const int* op,const int* q0,
 const int* q1,const double* angle,double lambda,double* probabilities,int workers){
 if(n<1 || n>12 || lambda<0 || lambda>1)return 1;
 const size_t N=size_t(1)<<n;
 std::vector<C> a(N*N,C(0)),b(N*N,C(0));a[0]=1;
 for(int g=0;g<count;++g){
  const int kind=op[g];if(kind==0)continue;
  size_t bit=size_t(1)<<q0[g],bit2=size_t(1)<<q1[g],mask=bit|bit2;
  C phase=std::exp(C(0,angle[g]));
  auto work=[&](size_t first,size_t last){
   for(size_t i=first;i<last;++i){
    size_t pi=i^((i&bit)?bit2:0),xi=i^bit,bi=i&~mask;
    for(size_t j=0;j<N;++j){
     size_t index=i*N+j;
     if(kind==1){ // RZ: U_i conjugate(U_j)
      int difference=int((i&bit)!=0)-int((j&bit)!=0);
      b[index]=a[index]*(difference==0?C(1):(difference>0?phase:std::conj(phase)));
     }else if(kind==2){ // SX, both matrix sides in a single pass
      size_t xj=j^bit;
      b[index]=.5*(a[index]+a[xi*N+xj])+C(0,.5)*(a[i*N+xj]-a[xi*N+j]);
     }else if(kind==3)b[index]=a[xi*N+(j^bit)];
     else if(kind==4){
      size_t pj=j^((j&bit)?bit2:0);
      C value=(1-lambda)*a[pi*N+pj];
      if(lambda && !((i^j)&mask)){
       size_t bj=j&~mask;
       value+=(lambda/4)*(a[bi*N+bj]+a[(bi|bit)*N+(bj|bit)]+a[(bi|bit2)*N+(bj|bit2)]+a[(bi|mask)*N+(bj|mask)]);
      }
      b[index]=value;
     }
    }
   }
  };
  int threads=std::max(1,std::min(workers,int(N)));std::vector<std::thread> jobs;
  for(int t=1;t<threads;++t)jobs.emplace_back(work,N*t/threads,N*(t+1)/threads);
  work(0,N/threads);for(auto& t:jobs)t.join();a.swap(b);
 }
 double trace=0;
 for(size_t i=0;i<N;++i){probabilities[i]=a[i*N+i].real();trace+=probabilities[i];}
 return std::abs(trace-1)>1e-7?2:0;
}
