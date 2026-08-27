#include <stdio.h>

void next_nop(int x, int y)
{
    if (x < 10)
    {
        printf("x is less than 10");
    }
    else
    {
        g(x, y);
    }
}
int main()
{
    int x = 6, y = 10;
    next_nop(x, y);
    return 0;
}