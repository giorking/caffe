//Message class function  file
//This file is define variable and functions of Message class
//Author: Xin Chen
//Novumind Inc.
//Version 1.0
//


#include "caffe/cluster/message.h"

//Construction of message
Message::Message( )
{
        m_iCommand = -1;
        m_iSource = -1;
        m_iTarget = -1;
        m_iLength = 0;
        m_pParam = NULL;
        m_Param1.m_iParam1 = 0;
        m_Param2.m_iParam2 = 0;
        m_Param3.m_iParam3 = 0;
}

Message::Message(int command, int source, int target, uint length, uchar* pParam)
{
        m_iCommand = command;
        m_iSource = source;
        m_iTarget = target;

        if (length > 0)
        {
                m_pParam = (uchar *)malloc(length*sizeof(char));
                if (!m_pParam)
                {
                       printf("Couldn't allocate memory in message construction 1\n");
                        exit(EXIT_FAILURE);
                }
               else
                {
                        if (pParam)
                                memcpy(m_pParam, pParam, length*sizeof(char));

                        m_iLength = length;
                }
        }
        else
        {
                m_iLength = 0;
                m_pParam = NULL;
        }
}
//


Message::~Message()
{
        Clear();
}
Message& Message::operator=(const Message& msg)
{
        m_iCommand = msg.m_iCommand;
        m_iSource = msg.m_iSource;
        m_iTarget = msg.m_iTarget;
        if( m_pParam != NULL )
        {
                free(m_pParam);
        }
        m_pParam = NULL;

        m_iLength = msg.m_iLength;

        if( m_iLength > 0 )
        {
                m_pParam = (unsigned char *)malloc(m_iLength);		
                if( msg.m_pParam)
                {
                        memcpy(m_pParam, msg.m_pParam, m_iLength);
                }
                else
                {
                        printf("Couldn't allocate memory in message\n");
                        exit(EXIT_FAILURE);
                }
        }

        m_Param1.m_iParam1 = msg.m_Param1.m_iParam1;
        m_Param2.m_iParam2 = msg.m_Param2.m_iParam2;
        m_Param3.m_iParam3 = msg.m_Param3.m_iParam3;


        return *this;
}

//Acclocate message buffer
uchar* Message::Buffer (uint length)
{
        if (!m_pParam)
                free (m_pParam);

        m_pParam = NULL;

        if (length)
        {
                m_pParam = (uchar *)malloc(length);
                if (!m_pParam)
                {
                        printf("Couldn't allocate memory in message at Buffer\n");
                        m_pParam = NULL;
                        m_iLength = 0;
                        exit(EXIT_FAILURE);
                }
                else
                {
                        m_iLength = length;
                }

        }
        else
        {
                m_iLength = 0;
        }
        return m_pParam;
}

void Message::Clear()
{
        if (m_pParam)
                free(m_pParam);

        m_pParam = NULL;
        m_iLength = 0;
        m_iCommand = -1;
        m_iSource = -1;
        m_iTarget = -1;
        m_Param1.m_iParam1 = 0;
        m_Param2.m_iParam2 = 0;
        m_Param3.m_iParam3 = 0;

}

char* getTime(int id)
{
        static char strtime[128];
        time_t now = time(0);
        tm *ltm = (tm*)localtime(&now);
        sprintf(strtime, "[%d/%d/%d/%d:%d:%d][%d] ", 1900+ltm->tm_year, 1+ltm->tm_mon, ltm->tm_mday,ltm->tm_hour, ltm->tm_min, ltm->tm_sec, id);
        return strtime;

}

char* getTime()
{
        static char strtime[128];
        time_t now = time(0);
        tm *ltm = (tm*)localtime(&now);
        sprintf(strtime, "[%d/%d/%d/%d:%d:%d] ", 1900+ltm->tm_year, 1+ltm->tm_mon, ltm->tm_mday,ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
        return strtime;

}
//Function: obtain command string name
//command: input
char * MsgStr( int command )
{
        static char strbuf[128];
        switch ( command )
        {
                case NN_INFINIBAND_READY:
                        strcpy(strbuf,"NN_INFINIBAND_READY");
                        break;
                case NN_INFINIBAND_REMINFO:
                        strcpy(strbuf,"NN_INFINIBAND_REMINFO");
                        break;
                case NN_FORWARDBACKWARD_START:
                        strcpy(strbuf,"NN_FORWARDBACKWARD_START");
                        break;
                default:
                        sprintf(strbuf,"No available message:%d ", command);
                        break;
        }
        return strbuf;
}
