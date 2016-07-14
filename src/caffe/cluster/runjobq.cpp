//Run JobQ file
//Author: Xin Chen
//May 26, 2016
//NovuMind Inc.
//Version 1.0
//
//
#include <glog/logging.h>
#include "caffe/cluster/runjobq.hpp"
RunJobQ::RunJobQ()
{
    if (pthread_mutex_init(&m_lock, NULL) != 0)
           LOG(FATAL)<<"Fail to create lock in runjobq";

}

RunJobQ::~RunJobQ()
{
        Clear();
        pthread_mutex_destroy(&m_lock);
}

void RunJobQ::Clear()
{
        Lock();
        while (!m_JobQueue.empty())
                m_JobQueue.pop();
        Unlock();
}

int RunJobQ::Lock()
{
        return pthread_mutex_lock(&m_lock);
}

int RunJobQ::Unlock()
{
        return pthread_mutex_unlock(&m_lock);
}

size_t RunJobQ::JobQsize()
{
        size_t l;
        Lock();
        l = m_JobQueue.size();
        Unlock();
        return l;
}

bool RunJobQ::JobQempty()
{
        bool b;
        Lock();
        b = m_JobQueue.empty();
        Unlock();
        return b;
}

uint RunJobQ::JobQpop()
{
        uint data;
        Lock();
        data = m_JobQueue.front();
        m_JobQueue.pop();
        Unlock();
        return data;
}

void RunJobQ::JobQpush(uint data)
{
        Lock();
        m_JobQueue.push(data);
        Unlock();
}
