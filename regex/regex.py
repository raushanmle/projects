import re
s = 'aaa@xxx.com bbb@yyy.com ccc@zzz.com ww.f333kart.com@ aaa@xyz.com'

a = re.search("a@",s)
a.span()
a[0]
s[2:4]
#match any seq of three digit
re.search('[0-9][0-9][0-9]', 'foo456bar')
#match any character in between 1 and 3 and (o-b)
re.search('1.3', 'foo1r3bar')
re.search('foo.bar', 'fooxbar')
print(re.search('foo.bar', 'foobar'))

"""
Character(s) - Meaning
.	Matches any single character except newline
^	Anchors a match at the start of a string
    Complements a character class
$	Anchors a match at the end of a string
*	Matches zero or more repetitions
+	Matches one or more repetitions
?	Matches zero or one repetition
    ∙ Specifies the non-greedy versions of *, +, and ?
    ∙ Introduces a lookahead or lookbehind assertion
    ∙ Creates a named group
{}	Matches an explicitly specified number of repetitions
\	∙ Escapes a metacharacter of its special meaning
    ∙ Introduces a special character class
    ∙ Introduces a grouping backreference
[]	Specifies a character class
|	Designates alternation
()	Creates a group
:   Designate a specialized group
#
=
!	Designate a specialized group

<>	Creates a named group

"""
#match first 2 fixed char and either of any char present in string
re.search('ba[artz]', 'foobarqux')
#match small case char from a to z
re.search('[a-z]', 'FOObar')
#match a to f both case and number 0 to 9
#Note: In the above examples, the return value is always the leftmost possible match. re.search() scans the search string from left to right, and as soon as it locates a match for <regex>, it stops scanning and returns the match.
re.search('[0-9a-fA-f]', '--- a0 ---')

# match number starting num and just after char
re.search('[^0-9]', '12345foo')
re.search('[#:^]', 'foo^bar:baz#qux')
# add slash to consider "-"
re.search('[ab\-c]', '123-456')
# we can keep special chars in []
re.search('[ab\]cd]', 'foo[1]')
re.search('[)*+|]', '123+456')

"""
--
\w matches any alphanumeric word character. Word characters are uppercase and lowercase letters, digits, and the underscore (_) character, so \w is essentially shorthand for [a-zA-Z0-9_]:
--
"""
re.search('\w', '#(.a$@&')

"""
\W upper case of W is the opposite. It matches any non-word character and is equivalent to [^a-zA-Z0-9_]
"""
re.search('\W', 'a_1*3Qb')

"""
\d matches any decimal digit character. \D is the opposite. It matches any character that isn’t a decimal digit:
"""
re.search('\d', 'abc4def')
re.search('\D', '234Q678')

# \s matches any whitespace character:
re.search('\s', 'foo\nbar baz')
# \S is the opposite of \s. It matches any character that isn’t whitespace:
re.search('\S', '  \n foo  \n  ')
"""
The character class sequences \w, \W, \d, \D, \s, and \S can appear inside a square bracket character class as well:
"""

re.search('[\d\w\s]', '---3--a-')
re.search('[\d\w\s]', '--- ---')

# escaping meta character
re.search('\.', 'foo.bar')

s = r'foo\bar'
re.search('\\\\', s)
re.search(r'\\', s)

# regex ^foo stipulates that 'foo' must be present not just any old place in the search string, but at the beginning: (^ or \A)

re.search('^foo', 'foobar') #foo word should be in starting
print(re.search('^foo', 'barfoo')) # returns none

"""
When the regex parser encounters $ or \Z, the parser’s current position must be at the end of the search string for it to find a match. Whatever precedes $ or \Z must constitute the end of the search string:
"""
re.search('bar$', 'foobar')
print(re.search('bar\Z', 'barfoo'))

"""
\b asserts that the regex parser’s current position must be at the beginning or end of a word. A word consists of a sequence of alphanumeric characters or underscores ([a-zA-Z0-9_]), the same as for the \w character class:
"""
>>> re.search(r'\bbar', 'foo bar')
<_sre.SRE_Match object; span=(4, 7), match='bar'>
>>> re.search(r'\bbar', 'foo.bar')
<_sre.SRE_Match object; span=(4, 7), match='bar'>

>>> print(re.search(r'\bbar', 'foobar'))

>>> re.search(r'foo\b', 'foo bar')
<_sre.SRE_Match object; span=(0, 3), match='foo'>
>>> re.search(r'foo\b', 'foo.bar')
<_sre.SRE_Match object; span=(0, 3), match='foo'>

>>> print(re.search(r'foo\b', 'foobar'))
re.search(r'\bbar\b', 'foo bar baz')

# \B does the opposite of \b. It asserts that the regex parser’s current position must not be at the start or end of a word:
re.search(r'\Bfoo\B', 'barfoobaz')
re.search('foo-*bar', 'foo--bar') #Two dashes
# .* matches everything between 'foo' and 'bar':
re.search('foo.*bar', '# foo $qux@grault % bar #')

# + Matches one or more repetitions of the preceding regex.
re.search('foo-+bar', 'foo--bar')

# ? Matches zero or one repetitions of the preceding regex.
re.search('foo-?bar', 'foobar')
re.match('foo[1-9]*bar', 'foo42bar')
print(re.match('foo[1-9]+bar', 'foo346bar'))
re.search('<.*>', '%<foo> <bar> <baz>%')
re.search('<.*?>', '%<foo> <bar> <baz>%')
re.search('ba?', 'baaaa')
# {m} Matches exactly m repetitions of the preceding regex.
re.search('x-{3}x', 'x---x')

# {m,n} Matches any number of repetitions of the preceding regex from m to n, inclusive.

# <regex>{,n}	Any number of repetitions of <regex> less than or equal to n

>>> for i in range(1, 6):
    s = f"x{'-' * i}x"
    print(f'{i}  {s:10}', re.search('x-{2,4}x', s))

re.search('x{foo,bar}y', 'x{foo,bar}y')
re.search('a{3,5}', 'aaaaaaaa')
re.search('a{3,5}?', 'aaaaaaaa')
re.search('(bar)', 'foo bar baz')
re.search('(bar)+', 'foo bar baz')
re.search('(bar)+', 'foo barbar baz')
re.search('bar+', 'foo barrrr baz') #matches only r any times

re.search('(ba[rz]){1,2}(qux)?', 'bazbarbazqux')
re.search('(foo(bar)?)+(\d\d\d)?', 'foofoobar123')

re.search('a+', 'aaaAAA', re.I)







re.sub("aaa","raus",s)
re.sub("a","raus",s,1)
re.sub("")
