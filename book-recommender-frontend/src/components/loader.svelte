<!-- src/lib/components/BookLoader.svelte -->
<script lang="ts">
	export let message = 'Finding your next great read';
	export let subtitle = 'Exploring our library';
</script>

<div class="fixed inset-0 z-50 flex items-center justify-center bg-base-100/80 backdrop-blur-sm">
	<div class="flex flex-col items-center space-y-6 bg-base-100 p-8 rounded-lg shadow-lg">
		<!-- Book animation container -->
		<div class="relative w-20 h-24">
			<!-- Book cover -->
			<div class="absolute inset-0 bg-slate-200 rounded shadow-lg animate-book-wiggle" />

			<!-- Book pages -->
			<div class="absolute right-0 w-1/2 h-full bg-base-100 rounded-r origin-left animate-page-flip">
				<!-- Page lines -->
				<div class="h-full p-2 flex flex-col justify-around">
					{#each Array(4) as _}
						<div class="h-0.5 bg-secondary rounded" />
					{/each}
				</div>
			</div>
		</div>

		<!-- Loading text -->
		<div class="text-center space-y-2">
			<p class="text-base-content font-medium">{message}</p>
			<p class="text-gray-500 text-sm">{subtitle}</p>
		</div>
	</div>
</div>

<style>
	
	@keyframes book-wiggle {
		0%,
		100% {
			transform: rotateY(0deg);
		}
		50% {
			transform: rotateY(-10deg);
		}
	}

	@keyframes page-flip {
		0%,
		100% {
			transform: rotateY(0deg);
		}
		50% {
			transform: rotateY(-120deg);
		}
	}

	:global(.animate-book-wiggle) {
		animation: book-wiggle 2s ease-in-out infinite;
	}

	:global(.animate-page-flip) {
		animation: page-flip 2s ease-in-out infinite;
	}
</style>
