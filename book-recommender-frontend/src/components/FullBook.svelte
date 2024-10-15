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
		image:
			'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
	};

	export let loading = true;
</script>

{#if loading}
	<div class="flex flex-row h-72 gap-10">
		<div class="flex flex-col w-64 p-3 text-center">
			<div class="skeleton h-full w-full"></div>
		</div>
		<div class="flex flex-col w-3/4 gap-5">
			<div class="skeleton h-10 w-28"></div>
			<div class="skeleton h-4 w-40"></div>
			<div class="flex flex-row text-lg align-middle">
				<div class="skeleton h-6 w-52"></div>
			</div>
			<div class="skeleton h-4 w-9/12"></div>
			<div class="skeleton h-4 w-full"></div>
			<div class="skeleton h-4 w-10/12"></div>
			<div class="skeleton h-4 w-11/12"></div>
			<div class="skeleton h-4 w-28"></div>
		</div>
	</div>
{:else}
	<div class="flex flex-row gap-10">
		<div class="flex flex-col w-64 p-3 text-center">
			<img src={book.image} alt={book.title} />
			<div class="m-3 mx-auto">
				<RatingStars></RatingStars>
			</div>

			<div class="text-lg font-medium text-base-content">Rate this book!</div>
		</div>
		<div class="flex flex-col w-3/4">
			<div class="stat">
				<div class="stat-value"><a href={book.infoLink} class="link text-wrap">{book.title}</a></div>
				<div class="stat-title text-xl mt-3">{book.author}</div>
				<div class="flex flex-row text-lg align-middle">
					<RatingStars id={1} interactible={false} rating={book.rating}></RatingStars>
					<span class="stat-value font-bold ml-10">{book.rating.toFixed(2)}</span>
					<span class="ml-3 my-auto">({book.countReviews} reviews)</span>
				</div>
				<div class="stat-desc whitespace-normal text-lg mt-8">{book.description}</div>
				<div class="mt-10 text-lg">Category: {book.category}</div>
			</div>
		</div>
	</div>
{/if}
