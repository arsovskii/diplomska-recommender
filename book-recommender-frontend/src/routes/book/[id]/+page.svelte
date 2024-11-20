<script>
	import { onMount } from 'svelte';
	import FullBook from '../../../components/FullBook.svelte';

	export let data;

	let loading = true;

	/**
	 * @type {any}
	 */
	let book_data;

	let book_id = data.bookId;
	const getBook = async () => {
		try {
			// Преземи ги податоците за книгата
			const response = await fetch('http://localhost:5000/api/book', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ book_id: book_id })
			});

			const text = await response.text();
			console.log('Raw response:', text);

			if (response.ok) {
				const data = JSON.parse(text);
				book_data = data.book;
				console.log('Book:', data);
			} else {
				console.error('Failed to fetch Book:', response.statusText);
			}
		} catch (error) {
			console.error('Error fetching book:', error);
		} finally {
			loading = false;
		}
	};

	onMount(async () => {
		await getBook();
	});
</script>

<div class="w-3/4 my-3 mx-auto">
	<FullBook book={book_data} {loading}></FullBook>
</div>
