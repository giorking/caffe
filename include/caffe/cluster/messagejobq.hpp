//Message JobQ header file
//Author: Xin Chen
//May 12, 2016
//Novumind Inc.
//Version 1.0
#ifndef MESSAGEJOBQ_HPP_
#define MESSAGEJOBQ_HPP_
#include "caffe/cluster/message.h"
#include <queue>
#include <pthread.h>

using namespace std;
class MessageJobQ
{
        public:
        private:
                pthread_mutex_t m_lock;
                queue<class Message> m_JobQueue;
        private:
                int Lock();//mutex locl
                int Unlock();//mutex unlock
        public:
                MessageJobQ();
                virtual ~MessageJobQ();
                void Clear(); //clear
                size_t JobQsize();//size of jobq
                bool JobQempty();
                void JobQpush(Message job);
                Message JobQpop();
                
};//end of messagejobq
#endif
