%{
#include <stdlib.h>
#define YYSTYPE double
#include "cal.tab.h"
%}

DIGIT [0-9]+
%option noyywrap

%%
{DIGIT} { yylval = atof(yytext); return NUM; }
\n|. { return yytext[0]; }
%%