#include <thread>
#include <chrono>
#include <iostream>
#include <cassert>

using namespace std;

void f1()
{
    cout << "f1() called\n";
}

void f2(int n)
{
    /* optional: make thread wait a bit */
    this_thread::sleep_for(chrono::milliseconds(10));
    cout << "f2() called with n = " << n << endl;
}

void f3(int &n)
{
    this_thread::sleep_for(chrono::milliseconds(20));
    cout << "f3() called; n is passed by reference; n = " << n << endl;
    n += 3;
}

void f4(int n, int &m)
{
    cout << "f4() called with n = " << n << " and m = " << m << endl;
    m += n;
}

int main(void)
{
    thread t1(f1);

    int m = 5;
    thread t2(f2, m);

    int k = 7;
    thread t3(f3, ref(k)); /* use std::ref to pass a reference */

    /* wait for all threads to finish */
    t1.join();
    t2.join();
    t3.join();

    cout << "k is now equal to " << k << endl;
    assert(k == 10);

    thread t4(f4, m, ref(k));
    t4.join();
    cout << "k is now equal to " << k << endl;
    assert(k == 15);

    return 0;
}
