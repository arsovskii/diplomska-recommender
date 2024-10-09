<script>
	import RatingStars from './ratingStars.svelte';

	export let book = {
		id: 0,
		title: 'Some Book',
		author: 'Avtor Avtoroski',
		image:
			'https://books.google.mk/books/content?id=pD6arNyKyi8C&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE73py6A4EznUkVRvEx7XRegXlsRbD8W0uZuBs9c6Gt_wx6UjUU0pF1th53Yx-bSK913V0zcmIHjdOEmB63-BU0AeBBt83rkLNhJIgDrP9teMEvCgB4aAPhWPBeB1PYXkbCIFOfP2'
	};
	
</script>

<div class="card card-compact aspect-[3/4] bg-base-100 shadow-xl w-64">
	<figure>
		<img src={book.image} alt={book.title} class="w-full h-70 object-cover" />
	</figure>
	<div class="card-body">
		<h2 class="card-title text-2xl font-bold">{book.title}</h2>
		<p class="text-l">{book.author}</p>
		<div class="card-actions justify-end">
			<RatingStars id={book.id}></RatingStars>
		</div>
	</div>
</div>
