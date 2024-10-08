<script>
	import { Slidy, Core } from '@slidy/svelte';
	import Book from './book.svelte';
	import { stairs } from '@slidy/animation';

	const slides = [
		{
			width: 800,
			height: 1200,
			id: 0,
			title: 'The Hobbit',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 1,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 2,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 3,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 4,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 5,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 6,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 7,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 8,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		},
		{
			width: 800,
			height: 1200,
			id: 9,
			title: 'The Lord of the Rings',
			author: 'J.R.R. Tolkien',
			image:
				'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
		}
	];

	let index = 1,
		position = 0,
		limit = 2;
</script>

<div class="parent">
	<Slidy
		{slides}
		let:item
		animation={stairs}
		axis="x"
		snap="center"
		groups="5"
		sensitivity="10"
		
	>
		<figure>
			<Book book={item}></Book>
		</figure>
	</Slidy>
</div>

<style>
	@import url('https://unpkg.com/@slidy/svelte/dist/slidy.css');

    .parent {
		--slidy-height:"7%";
		--slidy-slide-height:"70%"
	}
</style>
