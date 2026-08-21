%{
#include <ctype.h>
#include <stdio.h>

#define YYSTYPE double

int yylex(void);
int yyerror(const char *s);
%}

%token NUM
%left '+' '-'
%left '*' '/'
%right UMINUS

%%
Statment: E { printf("Answer: %g \n", $$); }
        | Statment '\n'
        ;

E : E '+' E { $$ = $1 + $3; }
  | E '-' E { $$ = $1 - $3; }
  | E '*' E { $$ = $1 * $3; }
  | E '/' E { $$ = $1 / $3; }
  | NUM
  ;
%%

int main() {
    printf("Enter the expression:\n");
    yyparse();
    return 0;
}

int yyerror(const char *s) {
    printf("%s\n", s);
    return 0;
}