<script>
	import RatingStars from "./ratingStars.svelte";

	export let book = {
		id: 0,
		title: 'Some Book',
		author: 'Avtor Avtoroski',
        description: 'nekakov opis',
        publisher: 'nekakov publisher',
        publishedDate: '2024-01-01',
        infoLink: 'https://books.google.mk/books?id=pD6arNyKyi8C&printsec=frontcover&dq=isbn:9781593275846&hl=&cd=1&source=gbs_api',
		category: 'nekakva',
		countReviews: 123,
		rating: 4.73,
		image:
			'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
	};

	export let loading = true;
</script>

{#if loading}
	<div class="skeleton w-32 h-32"></div>
{:else}
	<div class="flex flex-row">
        <div class="flex flex-col">
			<img src="{book.image}" alt="{book.title}">
            <RatingStars></RatingStars>
        </div>
        <div class="flex flex-col">
            <div>{book.title}</div>
            <div>{book.author}</div>
            <div>{book.description}</div>
        </div>
    </div>
{/if}
