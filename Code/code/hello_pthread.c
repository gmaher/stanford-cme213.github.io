#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include <pthread.h>

void ProcessOpt(int argc, char **argv, int *n_thread)
{
    int c;
    extern char *optarg;
    extern int optopt;
    *n_thread = 2;

    while ((c = getopt(argc, argv, "p:h")) != -1)
        switch (c)
        {
        case 'p':
            *n_thread = atoi(optarg);
            break;

        case 'h':
            fprintf(stderr, "Options:\n-p NTHREAD\tNumber of threads\n");
            exit(2);

        case '?':
            fprintf(stderr, "Unrecognized option: -%c\n", optopt);
            exit(2);
        }
}

long ComplexCalculation(long i)
{
    long a, c;
    a = 1103515245;
    c = 12345;
    i = (i * a + c) % 1000;
    return i;
}

void *PrintHello(void *threadid)
{
    long tid = (long)threadid;
    long result;
    result = ComplexCalculation(tid); /* Simulates some useful calculation */
    printf("Hello World! It's me, thread #%ld. My value is %ld!\n", tid,
           result);
    pthread_exit((void *)result /*NULL is common*/);
}

int main(int argc, char **argv)
{
    pthread_t *thread;
    int n_thread;
    long t;
    long thread_result;

    /* Command line options */
    ProcessOpt(argc, argv, &n_thread);
    assert(n_thread >= 1);

    thread = (pthread_t *)malloc(sizeof(pthread_t) * n_thread);

    for (t = 0; t < n_thread; t++)
    {
        printf("In main: creating thread %ld\n", t);
        pthread_create(&thread[t], NULL, PrintHello, (void *)t);
    }

    for (t = 0; t < n_thread; t++)
    {
        pthread_join(thread[t], (void **)(&thread_result) /*NULL is common*/);
        printf("Thread #%ld just finished; its value is %ld\n", t,
               thread_result);
        assert(ComplexCalculation(t) == thread_result); /* Testing output */
    }

    free(thread);

    printf("All tests passed. The master thread is exiting now.\n");
    return 0;
}
