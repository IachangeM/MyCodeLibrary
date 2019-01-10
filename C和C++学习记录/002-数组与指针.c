/*
from: 《C和指针》第八章 数组
@author:geekac
*/
#include <stdio.h>
#include <stdlib.h>

int main()
{
    /************  [Part One] 指针和数组的关系&指针运算   ******************/
    int array[3] = {1, 2, 3};

    int multi_dim_array[2][5][3] = {
        {
            {111, 112, 113},
            {121, 122, 123},
            {131, 132, 133},
            {141, 142, 143},
            {151, 152, 153},
        },
        {
            {211, 212, 213},
            {221, 222, 223},
            {231, 232, 233},
            {241, 242, 243},
            {251, 252, 253},
        }
    };

    printf("sizeof(int)              =  %lu\n", sizeof(int));

    printf("sizeof(array)            =  %lu\n", sizeof(array));
    printf("sizeof(multi_dim_array)  =  %lu\n", sizeof(multi_dim_array));

    printf("\n[address info]\n  pointer\t\tcontent\n");
    printf("  array \t\t%p\n",  array);
    printf("  array+1\t\t%p\n", array + 1);

    printf("  multi_dim_array\t%p\n",   multi_dim_array);
    printf("  multi_dim_array+1\t%p\n", multi_dim_array + 1);
    printf(" *multi_dim_array\t%p\n", * multi_dim_array);
    printf("**multi_dim_array\t%p\n", **multi_dim_array);

    /*
    Output/输出:
        sizeof(int)              =  4
        sizeof(array)            =  12
        sizeof(multi_dim_array)  =  120

        [address info]
          pointer		    content
          array 		    0x7fff695a7b30
          array+1		    0x7fff695a7b34
          multi_dim_array	0x7fff695a7b40
          multi_dim_array+1	0x7fff695a7b7c
         *multi_dim_array	0x7fff695a7b40
        **multi_dim_array	0x7fff695a7b40

    Explanation/解释：
        (1) 十六进制7c-40 = 60(二进制)
        (2) 在大多数表达式中，数组名是指向数组【第一个元素】的指针【常量】(不可以指向别的地址)。
            但是，有两个例外：
            a) sizeof(array_name)返回整个数组所占用的字节，而不是指针类型占用的字节数。
            b) &array_name返回指向数组的指针(数组指针)，而不是指向数组第一个元素的指针的指针。
        (3) multi_dim_array、*multi_dim_array、**multi_dim_array都是指向数组的指针，
            但是他们指向的数组类型不同。
            array是指向第一个元素的指针，也就是指向int类型的指针。
        (4) 指针做加减法运算时，指针变量的值的变化和其所指向的东西有关。
            特别注意指向数组的指针做加减法的情况。
            multi_dim_array指向的就是5×3的int类型的二维数组，所以：
            multi_dim_array+1转换为multi_dim_array+1*(5*3*4)，其值增加60。

    Tips/补充：
        (1) 两个指针相减的结果的类型是ptrdiff_t，是一种有符号类型。
            注意相减的结果是相差几个指针所指的类型，而不是简单的地址的值相减。
        (2) 如果两个做相减运算的指针不是同一个数组中的元素，其结果为未定义！

        (3) 指针可以做关系运算： > >=   < <=
        【重要】作为函数参数的数组名
            int strlen(char *string);
            int strlen(char string[]);
            等价于int strlen(char string[10]);//函数不为数组参数分配内存空间！！

            所有传递给函数的参数都是通过传值方式进行的，因此传递给函数的是该指针的一份拷贝。
            函数的参数是指针类型，并不是数组，因此执行sizeof(形参)无法得到数组的长度，
            而是指针类型占据的字节数。
            通常的做法是将数组长度通过形参传递。
    */


    /************  [Part Two] 指向数组的指针   ******************/

    int   *p1[10];//指针数组
    int *(p2)[10];//数组指针：指向数组的指针

    /*
    Knowledge/知识点：
        指针数组和数组指针的主要区别方法是 运算符的优先级：()>[]>*。
        p1先和[]结合构成数组的定义，"int *"表明数组元素的类型是int型指针；
        p2先和*结合构成指针的定义，int表明数组元素的类型，这里没有数组名(匿名数组)。
        并且p2的值等于数组的首地址。
        【重要】 声明数组指针时，如果省略数组的长度，则不能在该指针上执行任何指针运算。
                int (*p)[] = matrix;//运算时，将根据空数组的长度调整，也就是与零相乘！
    */


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
