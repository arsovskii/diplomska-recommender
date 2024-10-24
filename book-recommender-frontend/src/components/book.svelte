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
			'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
	};
</script>

<div class="book-card group w-64 transform-gpu flex-shrink-0" in:fade={{ duration: 300 }}>
	<div
		class="relative bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 active:scale-[0.98]"
	>
		<!-- Book cover with gradient overlay -->
		<div class="relative h-[60%] overflow-hidden rounded-t-xl">
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
			<div class="absolute top-2 left-2 z-20">
				<span
					class="px-2 py-1 text-xs font-medium text-white bg-black/50 backdrop-blur-sm rounded-full"
				>
					{book.category}
				</span>
			</div>
		</div>

		<!-- Book info -->
		<div class="p-4 h-[40%] flex flex-col justify-between">
			<div>
				<a href="book/{book.id}" class="block group-hover:text-blue-600 transition-colors">
					<h3 class="font-bold text-lg leading-tight line-clamp-2 mb-1">
						{book.title}
					</h3>
					<p class="text-sm text-gray-600">
						{book.author}
					</p>
				</a>
			</div>

			<!-- Rating section -->
			<div class="rating-container mt-2 p-2">
				<div class="flex items-center justify-between mb-1">
					<RatingStars id={book.id} rating={book.rating} />
				</div>
				<div class="text-sm text-gray-600 text-right">
					{book.countReviews} reviews
				</div>
				<div class="text-sm text-gray-600 text-right">
					Average: {book.rating.toFixed(2)}
				</div>
			</div>
		</div>
	</div>
</div>
