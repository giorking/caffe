//Ru JobQ header file
//Author: Xin Chen
//May 27, 2016
//Novumind Inc.
//Version 1.0

#ifndef RUNJOBQ_HPP_
#define RUNJOBQ_HPP_
#include <queue>
#include <pthread.h>
using namespace std;

class RunJobQ
{
        public:
        private:
                pthread_mutex_t m_lock;
                queue<uint> m_JobQueue;
        private:
                int Lock();
                int Unlock();
        public:
                RunJobQ();
                virtual ~RunJobQ();
                void Clear();
                size_t JobQsize();
                bool JobQempty();
                uint JobQpop();
                void JobQpush (uint data);

};
#endif




