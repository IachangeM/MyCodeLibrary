
#include <stdio.h>
#include <stdlib.h>

int main()
{

    int array2[][3] = {
        {1, 2, 3},
        {4, 5, 5}
    };
    
    //数组指针和
    printf("%p\n", array2);printf("%p\n", *array2);

    return 0;
    int array[2][5][3] = {
        {
            {11, 12, 13},
            {14, 15, 16},
            {17, 18, 19},
            {110, 111, 112},
            {113, 114, 115},
        },
        {
            {21, 22, 23},
            {24, 25, 26},
            {27, 28, 29},
            {210, 211, 212},
            {213, 214, 215},

        }

    };

    //printf("%d\n", **(*(array + 1) + 4));
    printf("%p\n",         *(array+1)+4
           );

    printf("%p\n",      *( *(array+1)+4  )
           );
    return 0;





    char const *keyword[] = {
        "do",
        "for",
        "if",
        "register",
        "zz",
        NULL,
    };

    printf("%s\n", keyword[3]);
    printf("total Bytes: %lu\n", sizeof(keyword));

    char const **str = NULL;
    for(str=keyword; *str != NULL; str++){
        printf("item's len=%lu\n", sizeof(str));
    }
    printf("items num: %lu\n", sizeof(keyword)/sizeof(keyword[0]));

    return 0;
}
