<script>
	import { fade } from 'svelte/transition';
	import RatingStars from './ratingStars.svelte';

	export let book = {
		id: 0,
		title: 'Some Book',
		author: 'Avtor Avtoroski',
		category: 'nekakva',
		countReviews: 123,
		rating: 4.73,
		image:
			'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2',
		prediction: 0.98
	};
</script>

<div class="book-card group w-64 h-[400px] transform-gpu flex-shrink-0" in:fade={{ duration: 300 }}>
	<div
		class="relative bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 active:scale-[0.98] h-full flex flex-col"
	>
		<!-- Book cover with gradient overlay -->
		<div class="relative h-[240px] overflow-hidden rounded-t-xl">
			<div
				class="absolute inset-0 bg-gradient-to-t from-black/40 via-black/0 to-black/10 z-10 opacity-60 group-hover:opacity-40 transition-opacity"
			></div>
			<img
				src={book.image}
				alt={book.title}
				class="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-500"
				loading="lazy"
			/>
			<!-- Category badge -->
			{#if book.category}
				<div class="absolute top-2 left-2 z-20">
					<span
						class="px-2 py-1 text-xs font-medium text-white bg-black/50 backdrop-blur-sm rounded-full"
					>
						{book.category}
					</span>
				</div>
			{/if}
		</div>
		<!-- Book info -->
		<div class="relative p-4 flex-1 flex flex-col justify-between">
			<div>
				<a
					href="book/{book.id}"
					class="block text-primary-content group-hover:text-secondary transition-colors"
				>
					<h3 class="font-bold text-lg leading-tight line-clamp-2 mb-1">
						{book.title}
					</h3>
					<p class="text-sm text-gray-600">
						{book.author}
					</p>
				</a>
			</div>
			<!-- Rating section -->
			<div class="rating-container mt-auto p-2">
				<div class="flex items-end justify-end mb-1">
					<RatingStars id={book.id} rating={book.rating} />
				</div>
				<div class="text-sm text-gray-600 text-right">
					{book.countReviews} reviews
				</div>
				<div class="text-sm text-gray-600 text-right">
					Average: {book.rating.toFixed(2)}
				</div>
			</div>
			{#if book.prediction}
				<div class="absolute bottom-0 left-0 p-3">
					<div class="badge badge-secondary text-sm text-secondary-content text-right">
						<div class="p-1">
							{(book.prediction * 100).toFixed(0)}% Match
						</div>
					</div>
				</div>
			{/if}
		</div>
	</div>
</div>
