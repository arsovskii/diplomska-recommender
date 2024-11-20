<script>
	import RatingStars from './ratingStars.svelte';

	export let book = {
		id: 0,
		title: 'Some Book',
		author: 'Avtor Avtoroski',
		description: 'nekakov opis',
		publisher: 'nekakov publisher',
		publishedDate: '2024-01-01',
		infoLink:
			'https://books.google.mk/books?id=pD6arNyKyi8C&printsec=frontcover&dq=isbn:9781593275846&hl=&cd=1&source=gbs_api',
		category: 'nekakva',
		countReviews: 123,
		rating: 4.73,
		image: undefined
	};
	export let loading = true;

	
	const formatDate = (/** @type {string | number | Date} */ dateString) => {
		if (!dateString) return null;
		const date = new Date(dateString);
		return !isNaN(date.getTime()) ? date.toLocaleDateString() : null;
	};
</script>

{#if loading}
	<div
		class="flex flex-col md:flex-row min-h-[24rem] gap-8 p-6 bg-base-100 rounded-lg shadow-lg animate-pulse"
	>
		<div class="flex flex-col w-full md:w-64 items-center">
			<div class="skeleton h-80 w-56"></div>
			<div class="skeleton h-8 w-40 mt-4"></div>
			<div class="skeleton h-6 w-32 mt-2"></div>
		</div>
		<div class="flex flex-col flex-1 gap-6">
			<div class="skeleton h-12 w-3/4"></div>
			<div class="skeleton h-6 w-48"></div>
			<div class="skeleton h-8 w-64"></div>
			<div class="skeleton h-4 w-full"></div>
			<div class="skeleton h-4 w-full"></div>
			<div class="skeleton h-4 w-3/4"></div>
			<div class="skeleton h-6 w-40 mt-4"></div>
		</div>
	</div>
{:else}
	<div
		class="flex flex-col md:flex-row gap-8 p-6 bg-base-100 rounded-lg shadow-lg transition-all hover:shadow-xl"
	>
		<div class="flex flex-col w-full md:w-64 items-center">
			<div class="relative group">
				<img
					src={book.image}
					alt={book.title}
					class="w-56 h-auto rounded-lg shadow-md transition-transform duration-300 group-hover:scale-105"
				/>
				<div
					class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all duration-300 rounded-lg"
				></div>
			</div>

			<div class="mt-6 space-y-3 text-center">
				<RatingStars id={book.id} />
				<div class="text-lg font-medium text-primary-content">Rate this book!</div>
			</div>
		</div>

		<div class="flex flex-col flex-1">
			<div class="space-y-4">
				<h1 class="text-3xl font-bold hover:text-primary transition-colors">
					<a href={book.infoLink} class="hover:underline">{book.title}</a>
				</h1>

				{#if book.author}
					<h2 class="text-xl text-base-content/80">by {book.author}</h2>
				{/if}

				<div class="flex flex-wrap items-center gap-4 mt-2">
					<RatingStars id={1} interactible={false} rating={book.rating} />
					<span class="text-2xl font-bold text-primary drop-shadow-[0_5px_5px_rgba(0,0,0,0.2)] ">{book.rating.toFixed(2)}</span>
					<span class="text-base-content/70">
						({book.countReviews?.toLocaleString() ?? 0} reviews)
					</span>
				</div>

				{#if book.description}
					<div class="mt-6 space-y-4">
						<p class="text-lg leading-relaxed text-base-content/90">{book.description}</p>
					</div>
				{/if}

				<div class="flex flex-wrap gap-4 pt-4">
					{#if book.category}
						<div class="badge badge-primary badge-lg">
							{book.category}
						</div>
					{/if}

					{#if formatDate(book.publishedDate)}
						<div class="badge badge-ghost badge-lg">
							Published: {formatDate(book.publishedDate)}
						</div>
					{/if}

					{#if book.publisher}
						<div class="badge badge-ghost badge-lg">
							{book.publisher}
						</div>
					{/if}
				</div>
			</div>
		</div>
	</div>
{/if}
