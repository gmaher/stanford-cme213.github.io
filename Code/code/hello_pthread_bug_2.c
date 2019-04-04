#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include <pthread.h>

void ProcessOpt(int argc, char** argv, int* n_thread) {
    int c;
    extern char *optarg;
    extern int optopt;
    *n_thread = 2;

    while((c = getopt(argc, argv, "p:h")) != -1)
        switch(c) {
        case 'p':
            *n_thread = atoi(optarg);
            break;

        case 'h':
            fprintf(stderr,"Options:\n-p NTHREAD\tNumber of threads\n");
            exit(2);

        case '?':
            fprintf(stderr,"Unrecognized option: -%c\n", optopt);
            exit(2);
        }
}

void DelayThread(int r) {
    struct timespec req, rem;
    req.tv_sec = 0;
    req.tv_nsec = r;
    /* Wait time in nanoseconds*/
    nanosleep(&req, &rem);
}

#define UNITTIME 100000

void Init(long i) {
    srand(i*1234*4096 + 2018); /* This resets the seed of rand() */
}

long Compute() {
    return (long)(rand() % 1000);
}

void* RaceConditionError(void* threadid) {
    long tid = (long) threadid;
    
    long r_wait_0 = 1+rand()%1000;
    long r_wait_1 = 1+rand()%1000;
    
    DelayThread(r_wait_0*UNITTIME);
    /* This line randomizes which thread is first */
    printf("Thread #%ld: Init()\n", tid);
    Init(tid); /* Resets the random seed */
    DelayThread(r_wait_1*UNITTIME);
    /* This line causes the race condition */
    printf("Thread #%ld: Compute()\n", tid);
    long result = Compute();
    /* Which seed is used is random at this stage */
    pthread_exit((void*)result);
}

int main(int argc, char** argv) {
    pthread_t* thread;
    int n_thread;
    int rc;
    long t;
    long * thread_result;
    
    /* initialize random seed */
    srand(time(NULL));

    /* Command line options */
    ProcessOpt(argc, argv, &n_thread);
    assert(n_thread >= 1);

    thread = (pthread_t*) malloc(sizeof(pthread_t) * n_thread);

    for(t = 0; t < n_thread; t++) {
        printf("In main: creating thread %ld\n", t);
        rc = pthread_create(&thread[t], NULL, RaceConditionError, (void*)t);

        if(rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(2);
        }
    }

    thread_result = (long*) malloc(sizeof(long) * n_thread);
    for(t = 0; t < n_thread; t++) {
        pthread_join(thread[t], (void**)(thread_result+t));
    }
    for(t = 0; t < n_thread; t++) {
        Init(t);
        long correct = Compute();
        printf("[main] Thread %2ld; ",t);
        if (correct == thread_result[t])
            printf("correct!\n");
        else
            printf("error! The correct result is %3ld [thread got %3ld]\n", correct, thread_result[t]);
    }

    free(thread_result);
    free(thread);

    return 0;
}
