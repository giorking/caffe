//Message JobQ file //Author: Xin Chen
//b
//May 12, 2016
//NovuMind Inc.
//Version 1.0

#include <glog/logging.h>

#include "caffe/cluster/messagejobq.hpp"
//MessageJobQ Constructor
MessageJobQ::MessageJobQ()
{
        if (pthread_mutex_init(&m_lock, NULL) != 0)
                LOG(FATAL)<<"Fail to create lock in MessageJobQ";
}


//MessageJobQ deconstructor
MessageJobQ::~MessageJobQ()
{
        Clear();
        pthread_mutex_destroy(&m_lock);

}

//Lock function
//Return: 0 sucess, others: error number
int MessageJobQ::Lock()
{
        return pthread_mutex_lock(&m_lock);
}

//unlock function
//Return: 0 sucess, others: error number
int MessageJobQ::Unlock()
{
        return pthread_mutex_unlock(&m_lock);
}

void MessageJobQ::Clear()
{
        Lock();
        while (!m_JobQueue.empty())
                m_JobQueue.pop();
        Unlock();
}

size_t MessageJobQ::JobQsize()
{
        size_t l;
        Lock();
        l = m_JobQueue.size();
        Unlock();
        return l;
}

bool MessageJobQ::JobQempty()
{
        bool b;
        Lock();
        b = m_JobQueue.empty();
        Unlock();
        return b;
}

Message MessageJobQ::JobQpop()
{
        Message data;
        Lock();
        data = m_JobQueue.front();
        m_JobQueue.pop();
        Unlock();   
        return data;
}

void MessageJobQ::JobQpush(Message data)
{
        Lock();
        m_JobQueue.push(data);
        Unlock();
}
