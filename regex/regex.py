# Author: Raushan Kumar

# importing re module
import re

############## character cheat sheet ###############

"""
=> .: Matches any single character except newline. E.g., a.b matches "acb", "a2b", "a$b", etc.
=> ^: Anchors a match at the start of a string. E.g., ^abc matches "abc" at the start of a string.
=> $: Anchors a match at the end of a string. E.g., abc$ matches "abc" at the end of a string.
=> *: Matches zero or more repetitions of the preceding character or group. E.g., a* matches "", "a", "aa", "aaa", etc.
=> +: Matches one or more repetitions of the preceding character or group. E.g., a+ matches "a", "aa", "aaa", etc., but not "".
=> ?: Matches zero or one repetition of the preceding character or group. E.g., a? matches "" and "a".
=> {n}: Matches exactly n repetitions of the preceding character or group. E.g., a{3} matches "aaa".
=> {n,}: Matches n or more repetitions of the preceding character or group. E.g., a{2,} matches "aa", "aaa", "aaaa", etc.
=> {n,m}: Matches between n and m repetitions of the preceding character or group. E.g., a{2,3} matches "aa" and "aaa".
=> \: Escapes a metacharacter of its special meaning, introduces a special character class, or introduces a grouping backreference.
=> []: Specifies a character class. E.g., [abc] matches "a", "b", or "c".
=> |: Designates alternation. E.g., a|b matches "a" or "b".
=> (): Creates a group. E.g., (abc) groups "abc" as a single unit.
=> \d: Matches any decimal digit; equivalent to [0-9].
=> \D: Matches any non-digit character; equivalent to [^0-9].
=> \s: Matches any whitespace character; equivalent to [ \t\n\r\f\v].
=> \S: Matches any non-whitespace character; equivalent to [^ \t\n\r\f\v].
=> \w: Matches any alphanumeric character; equivalent to [a-zA-Z0-9_].
=> \W: Matches any non-alphanumeric character; equivalent to [^a-zA-Z0-9_].
=> \b: Matches where the specified characters are at the beginning or end of a word.
=> \B: Matches where the specified characters are present, but NOT at the beginning (or end) of a word.

"""
##################   Cheat Sheet End ##################

s = "Hello, my name is John Doe. I live in New York, USA. My email address is john.doe@example.com. I was born on 1985-05-15. I work at XYZ Corp. My website is http://www.johndoe.com. My phone number is +1-555-555-5555. I have a dog named Fido. I love to play football and basketball. My favorite numbers are 7 and 13. I often say 'Hello, World!' when I'm coding. My favorite programming languages are Python and JavaScript. I have a Master's degree in Computer Science. My favorite book is 'To Kill a Mockingbird'. I drive a Toyota Camry. My favorite color is blue. I like to eat pizza and hamburgers. I go to bed at 11:00 PM and wake up at 7:00 AM. I like to listen to music by The Beatles. I have visited Paris, London, and Tokyo. I have a credit card with the number 1234-5678-9012-3456. My social security number is 123-45-6789. "

# Let's take a sample string for all examples
s = 'aaa@xxx.com bbb@yyy.com ccc@zzz.com ww.f333kart.com@ aaa@xyz.com'

# searching specific string
a = re.search("@", s)
# query string occurance
a.span()
# what you queried
a[0]
# specific part of string
s[2:4]
# match any seq of three digit
re.search('[0-9][0-9][0-9]', s)
# match any character in between 1 and 3 and (o-b)
re.search('1.3', s)
re.search('d.*@', s)
# match first 2 fixed char and either of any char present in string
re.search('ba[artz]', 'foobarqux')
# match small case char from a to z
re.search('[a-z]', s)

re.search('[d].*@', s)
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

#  \w : matches any alphanumeric word character. Word characters are uppercase and lowercase letters, digits, and the underscore (_) character, so \w is essentially shorthand for [a-zA-Z0-9_]:

re.search('\w', '#(.a$@&')

# \W : It's opposite of w. It matches any non-word character and is equivalent to [^a-zA-Z0-9_]

re.search('\W', 'a_1*3Qb')

# \d matches any decimal digit character. \D is the opposite. It matches any character that isn’t a decimal digit:
re.search('\d', 'abc4def')
re.search('\D', '234Q678')

# \s matches any whitespace character:
re.search('\s', 'foo\nbar baz')

# \S is the opposite of \s. It matches any character that isn’t whitespace:
re.search('\S', '  \n foo  \n  ')

# The character class sequences \w, \W, \d, \D, \s, and \S can appear inside a square bracket character class as well:
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

# When the regex parser encounters $ or \Z, the parser’s current position must be at the end of the search string for it to find a match. Whatever precedes $ or \Z must constitute the end of the search string:

re.search('bar$', 'foobar')
print(re.search('bar\Z', 'barfoo'))

# \b asserts that the regex parser’s current position must be at the beginning or end of a word. A word consists of a sequence of alphanumeric characters or underscores ([a-zA-Z0-9_]), the same as for the \w character class:

re.search(r'\bbar', 'foo.bar')
print(re.search(r'\bbar', 'foobar'))
re.search(r'foo\b', 'foo bar')
re.search(r'foo\b', 'foo.bar')
print(re.search(r'foo\b', 'foobar'))
re.search(r'\bbar\b', 'foo bar baz')
re.search(r'\bbar\b', 'bar foo baz')

# \B does the opposite of \b. It asserts that the regex parser’s current position must not be at the start or end of a word:
re.search(r'\Bfoo\B', 'barfoobaz')

# .* matches everything between 'foo' and 'bar':
re.search('foo-*bar', 'foo--bar')
# .* matches everything between 'foo' and 'bar':
re.search('foo.*bar', '# foo $qux@grault % bar #')

# + Matches one or more repetitions of the preceding regex.
re.search('foo-+bar', 'foo--bar')
re.search('foo+bar', 'foooooooooobar')

# ? Matches zero or one repetitions of the preceding regex.
re.search('foo-?bar', 'foobar')
re.search('foo-?bar', 'foo-bar')
re.match('foo[1-9]*bar', 'foo467672bar')
re.match('foo[1-9]+bar', 'foo467672bar')
re.search('<.*>', '%<foo> <bar> <baz>%')
re.search('<.*?>', '%<foo> <bar> <baz>%')
re.search('ba?', 'baaaa')

# {m} Matches exactly m repetitions of the preceding regex.
re.search('x-{3}x', 'x---x')

# {m,n} Matches any number of repetitions of the preceding regex from m to n, inclusive.
re.search('x-{1,3}x', 'x-----------x')
re.search('x-{1,3}x', 'x--x')
# <regex>{,n}	Any number of repetitions of <regex> less than or equal to n
re.search('x-{,3}x', 'x------x')
re.search('x-{,3}x', 'x--x')
re.search('x-{,3}x', 'xx')

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

################ sub #####################
s = 'aaa@xxx.com bbb@yyy.com ccc@zzz.com ww.f333kart.com@ aaa@xyz.com'

re.sub("aaa","raus", s)
re.sub("a", "raus", s, 1)
re.sub("")

string1 = "Hello, world. hey word \n"
re.search(r"....", string1)
re.search(r"l+o", string1)
re.search(r"H.?e", string1) # 0 or 1 char in between H and e

# Matches the preceding element zero or more times. For example, ab*c matches "ac", "abc", "abbbc", etc. [xyz]* matches "", "x", "y", "z", "zx", "zyx", "xyzzy", and so on. (ab)* matches "", "ab", "abab", "ababab", and so on.

re.search(r"e(ll)*o", string1) #(ab) matches ababab only in this pattern any times

re.search(r"^He", string1) #begins with
re.search(r"rld$", string1) #ends with

# filter sindwidtched between vowles
re.search(r"[aeioul]+", string1)
# does not start with H a b c
re.search(r"[^Habc]", string1)
re.search(r"....[d]", string1) #match 4 chars before "."

re.findall("@...........", s)

#substitute with value # last number how many time replacement shd be performed
print(re.sub('[a-z]*@', 'ABC@', s,3))

# any item matched will be replaced
print(re.sub('[xyz23]', '1', s))
print(re.sub('aaa|bbb|ccc', 'ABC', s))
print(re.sub('([a-z]*)@', '\\1-123@', s))

re.subn('[a-z]*@', 'ABC@', s)


re.sub('\d{3}', 'ABC', s) # replace that many times repeated digits to char
re.sub('\d', 'ABC', s)
