#ifndef LIST_H
#define LIST_H
#include "darknet.h"
// 作者的list 函数

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);


void free_list_contents(list *l);

#endif
