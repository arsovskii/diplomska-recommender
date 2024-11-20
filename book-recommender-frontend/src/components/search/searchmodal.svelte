<script lang="ts">
	import SearchTable from './searchTable.svelte';

	let timer: number | undefined;
	let value = '';
	let newTitle = '';

	let books: never[] = [];
	let loading: boolean = false;
	let inputting: boolean = false;

	// фунцкија за пребарување на книга, со debounce - при промена на вредноста на input полето, се чека 200ms пред да се изврши пребарувањето
	function searchBook() {
		console.log('searching...');
		clearTimeout(timer);

		loading = true;
		timer = setTimeout(() => {
			value = newTitle;
			console.log(value);
			if (value.length > 0) {
				inputting = true;
				getSearchedBooks();
			} else {
				inputting = false;
			}
		}, 200);
	}

	// фунцкија за пребарување на книга
	const getSearchedBooks = async () => {
		try {
			const response = await fetch('http://localhost:5000/api/search', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ title: value })
			});

			const text = await response.text();
			console.log('Raw response:', text);

			if (response.ok) {
				const data = JSON.parse(text);
				books = data.books;
				loading = false;

				console.log('Recommendations:', books);
			} else {
				console.error('Failed to fetch searched books:', response.statusText);
			}
		} catch (error) {
			console.error('Error fetching searched books:', error);
		}
	};

	let search_modal: HTMLDialogElement;

	function closeModal() {
		search_modal.close();
	}
</script>

<dialog id="search_modal" class="modal h-full overflow-hidden" bind:this={search_modal}>
	<div class="modal-box mt-0 pt-0">
		<form method="dialog sticky">
			<button class="btn btn-sm btn-circle btn-ghost absolute right-2 top-2 z-20" on:click={closeModal}>✕</button>
		</form>
		<div
			class="text-lg font-bold sticky top-0 py-6 z-10 bg-gradient-to-b from-base-100 from-60% via-base-100 to-transparent"
		>
			<h3 class="text-lg font-bold mb-3 pb-3">
				Search for a book
				
			</h3>
			<div class="w-full my-3">
				<input
					type="text"
					placeholder="Type here"
					class="input input-bordered input-primary w-full"
					bind:value={newTitle}
					on:input={searchBook}
				/>
			</div>
		</div>
		<SearchTable {books} onLinkClick={closeModal} {loading} {inputting}></SearchTable>
	</div>
</dialog>
