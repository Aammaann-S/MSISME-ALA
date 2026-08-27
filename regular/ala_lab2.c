#include <stdio.h>

void creat_handle(char id, char bank_id, char account_no)
{
    if (id != "" && id != "")
    {
        printf(id, "@", bank_id);
    }
    else
    {
        printf("Data Missing");
    }
}

int main()
{
    int x = "abcd", y = "bankname";
    creat_handle(x, y, "");
    return 0;
}