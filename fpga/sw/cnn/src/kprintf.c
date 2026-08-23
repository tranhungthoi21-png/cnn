// See LICENSE.Sifive for license details.
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>

#include "kprintf.h"

static inline void _kputs(const char *s)
{
	char c;
	for (; (c = *s) != '\0'; s++)
		kputc(c);
}

void kputs(const char *s)
{
	_kputs(s);
	kputc('\r');
	kputc('\n');
}

void kprintf(const char *fmt, ...)
{
	va_list vl;
	bool is_format, is_long, is_char;
	char c;

	va_start(vl, fmt);
	is_format = false;
	is_long = false;
	is_char = false;
	while ((c = *fmt++) != '\0') {
		if (is_format) {
			switch (c) {
			case 'l':
				is_long = true;
				continue;
			case 'h':
				is_char = true;
				continue;
			case 'd':
			case 'u': {
				long n;
				if (is_long) {
					n = va_arg(vl, long);
				} else {
					n = va_arg(vl, int);
				}

				if (c == 'd' && n < 0) {
					kputc('-');
					n = -n;
				}

				if (n == 0) {
					kputc('0');
					break;
				}

				char buf[32];
				int i = 0;
				while (n > 0) {
					buf[i++] = '0' + (n % 10);
					n /= 10;
				}
				while (i > 0) {
					kputc(buf[--i]);
				}
				break;
			}
			case 'x':
			case 'p': {
				unsigned long n;
				long i;
				if (is_long || c == 'p') {
					n = va_arg(vl, unsigned long);
					i = (sizeof(unsigned long) << 3) - 4;
				} else {
					n = va_arg(vl, unsigned int);
					i = is_char ? 4 : (sizeof(unsigned int) << 3) - 4;
				}
				if (c == 'p') {
					_kputs("0x");
				}
				for (; i >= 0; i -= 4) {
					long d;
					d = (n >> i) & 0xF;
					kputc(d < 10 ? '0' + d : 'a' + d - 10);
				}
				break;
			}
			case 's': {
				const char *s = va_arg(vl, const char *);
				if (!s) s = "(null)";
				_kputs(s);
				break;
			}
			case 'c':
				kputc(va_arg(vl, int));
				break;
			case '%':
				kputc('%');
				break;
			}
			is_format = false;
			is_long = false;
			is_char = false;
		} else if (c == '%') {
			is_format = true;
		} else {
			kputc(c);
		}
	}
	va_end(vl);
}
