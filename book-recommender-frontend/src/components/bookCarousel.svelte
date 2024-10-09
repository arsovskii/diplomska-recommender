<script lang="ts">
	import { Slidy, Core, classNames } from '@slidy/svelte';
	import Book from './book.svelte';
	import { stairs, translate } from '@slidy/animation';
	import { fade } from 'svelte/transition';
	import anime from 'animejs';

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

	let hasMounted: Boolean = false;
	let slidyMount = () => {
		hasMounted = true;

		let allSlidy = document.getElementsByClassName('slidy-slide');
		for (let i = 0; i < allSlidy.length; i++) {
			(allSlidy[i] as HTMLElement).style.opacity = '0';
		}

		document.getElementById('slidy-container')!.style.display = 'block';

		anime({
			targets: '.slidy-slide',
			opacity: 1,
			duration:200,
			delay: anime.stagger(50) // increase delay by 100ms for each elements.
		});
	};
</script>

<div transition:fade id="slidy-container" class="transition-all" style="display:none;">
	<Slidy
		{slides}
		let:item
		animation={stairs}
		axis="x"
		snap="center"
		sensitivity="10"
		--slidy-counter-bg="oklch(var(--s))"
		--slidy-arrow-bg="oklch(var(--s))"
		
		
		on:mount={slidyMount}
	>
		<figure >
			<Book book={item}></Book>
		</figure>
	</Slidy>
</div>
{#if !hasMounted}
	<div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-1/2 my-20 text-center">
		<div class="m-auto">
			<span class="loading loading-dots w-20"></span>
			<div>Интересен факт: ние никогаш не грешиме!</div>
		</div>
	</div>
{/if}

<style>
	@import url('https://unpkg.com/@slidy/svelte/dist/slidy.css');
	
</style>
