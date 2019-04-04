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

long ComplexCalculation(long i) {
    long a, c;
    a = 1103515245;
    c = 12345;
    i = (i*a+c) % 1000;
    return i;
}

void* PrintHello(void* threadid) {
    long tid = (long)threadid;
    long result;
    result = ComplexCalculation(tid); /* Simulates some useful calculation */
    printf("Hello World! It's me, thread #%ld. My value is %ld!\n", tid, result);
    pthread_exit((void*)(&result));
}

int main(int argc, char** argv) {
    pthread_t* thread;
    int n_thread;
    int rc;
    long t;
    long* p_thread_result;

    /* Command line options */
    ProcessOpt(argc, argv, &n_thread);
    assert(n_thread >= 1);

    thread = (pthread_t*) malloc(sizeof(pthread_t) * n_thread);

    for(t = 0; t < n_thread; t++) {
        printf("In main: creating thread %ld\n", t);
        rc = pthread_create(&thread[t], NULL, PrintHello, (void*)t);

        if(rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            exit(2);
        }
    }

    for(t = 0; t < n_thread; t++) {
        pthread_join(thread[t], (void**)(&p_thread_result)/*NULL is common*/);
        printf("Thread #%ld just finished; its value is %ld\n", t, *p_thread_result);
        printf("The previous line is expected to make the code crash.\n");
        printf("If the previous line was executed, at least the number should be wrong.\n");
    }

    free(thread);
    return 0;
}
